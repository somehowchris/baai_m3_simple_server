from FlagEmbedding import BGEM3FlagModel
from typing import List, Tuple, Union, cast
import asyncio
from fastapi import FastAPI, Request, Response, HTTPException
from starlette.status import HTTP_504_GATEWAY_TIMEOUT
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from uuid import uuid4
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
import pydantic_numpy.typing as pnd
import argparse
import torch

# Set up command line argument parsing
parser = argparse.ArgumentParser(description="Run the FastAPI server for embeddings and re-ranking")
parser.add_argument('--batch_size', type=int, default=128, help='GPU batch size')
parser.add_argument('--max_request', type=int, default=10, help='Maximum number of requests to process')
parser.add_argument('--max_length', type=int, default=8192, help='Maximum context length for embeddings and passages')
parser.add_argument('--max_q_length', type=int, default=8192, help='Maximum context length for questions')
parser.add_argument('--request_flush_timeout', type=float, default=0.1, help='Request flush timeout')
parser.add_argument('--rerank_weights', type=float, nargs=3, default=[0.4, 0.2, 0.4], help='Re-rank score weights')
parser.add_argument('--request_time_out', type=int, default=30, help='Request timeout threshold')
parser.add_argument('--gpu_time_out', type=int, default=10, help='GPU processing timeout threshold')
parser.add_argument('--port', type=int, default=3000, help='Port to run the server on')
parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device to run the model on')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model for faster inference')

args = parser.parse_args()

# Use the parsed arguments
batch_size = args.batch_size
max_request = args.max_request
max_length = args.max_length
max_q_length = args.max_q_length
request_flush_timeout = args.request_flush_timeout
rerank_weights = args.rerank_weights
request_time_out = args.request_time_out
gpu_time_out = args.gpu_time_out
port = args.port
device = args.device
compile = args.compile


class m3Wrapper:
    def __init__(self, model_name: str, device: str = "cuda", compile=False):
        self.model = BGEM3FlagModel(
            model_name, device=device, use_fp16=True if device != "cpu" else False
        )
        if compile == True:
            self.model.model = torch.compile(self.model.model)

    def embed(self, sentences: List[str]) -> List[tuple[Any, Any]]:
        emb = self.model.encode(
            sentences,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )
        embeddings = map(
            lambda x: {"dense": x[0], "colbert": x[1], "sparse": x[2]},
            zip(
                emb["dense_vecs"],
                emb["colbert_vecs"],
                emb["lexical_weights"],
            ),
        )
        
        return list(embeddings)

    def rerank(self, sentence_pairs: List[Tuple[str, str]]) -> List[float]:
        scores = self.model.compute_score(
            sentence_pairs,
            batch_size=batch_size,
            max_query_length=max_q_length,
            max_passage_length=max_length,
            weights_for_different_modes=rerank_weights,
        )["colbert+sparse+dense"]
        return scores


class EmbedRequest(BaseModel):
    sentences: List[str]


class RerankRequest(BaseModel):
    sentence_pairs: List[Tuple[str, str]]

class EmbedResponseItem(BaseModel):
    dense: List[float]
    colbert: List[List[float]]
    sparse: dict[str, float]

class EmbedResponse(BaseModel):
    embeddings: List[EmbedResponseItem]


class RerankResponse(BaseModel):
    scores: List[float]


class RequestProcessor:
    def __init__(
        self, model: m3Wrapper, max_request_to_flush: int, accumulation_timeout: float
    ):
        self.model = model
        self.max_batch_size = max_request_to_flush
        self.accumulation_timeout = accumulation_timeout
        self.queue = asyncio.Queue()
        self.response_futures = {}
        self.processing_loop_task = None
        self.processing_loop_started = False  # Processing pool flag lazy init state
        self.executor = ThreadPoolExecutor()  # Thread pool
        self.gpu_lock = asyncio.Semaphore(1)  # Sem for gpu sync usage

    async def ensure_processing_loop_started(self):
        if not self.processing_loop_started:
            print("starting processing_loop")
            self.processing_loop_task = asyncio.create_task(self.processing_loop())
            self.processing_loop_started = True

    async def processing_loop(self):
        while True:
            requests, request_types, request_ids = [], [], []
            start_time = asyncio.get_event_loop().time()

            while len(requests) < self.max_batch_size:
                timeout = self.accumulation_timeout - (
                    asyncio.get_event_loop().time() - start_time
                )
                if timeout <= 0:
                    break

                try:
                    req_data, req_type, req_id = await asyncio.wait_for(
                        self.queue.get(), timeout=timeout
                    )
                    requests.append(req_data)
                    request_types.append(req_type)
                    request_ids.append(req_id)
                except asyncio.TimeoutError:
                    break

            if requests:
                await self.process_requests_by_type(
                    requests, request_types, request_ids
                )

    async def process_requests_by_type(self, requests, request_types, request_ids):
        tasks = []
        for request_data, request_type, request_id in zip(
            requests, request_types, request_ids
        ):
            if request_type == "embed":
                task = asyncio.create_task(
                    self.run_with_semaphore(
                        self.model.embed, request_data.sentences, request_id
                    )
                )
            else:  # 'rerank'
                task = asyncio.create_task(
                    self.run_with_semaphore(
                        self.model.rerank, request_data.sentence_pairs, request_id
                    )
                )
            tasks.append(task)
        await asyncio.gather(*tasks)

    async def run_with_semaphore(self, func, data, request_id):
        async with self.gpu_lock:  # Wait for sem
            future = self.executor.submit(func, data)
            try:
                result = await asyncio.wait_for(
                    asyncio.wrap_future(future), timeout=gpu_time_out
                )
                self.response_futures[request_id].set_result(result)
            except asyncio.TimeoutError:
                self.response_futures[request_id].set_exception(
                    TimeoutError("GPU processing timeout")
                )
            except Exception as e:
                self.response_futures[request_id].set_exception(e)

    async def process_request(
        self, request_data: Union[EmbedRequest, RerankRequest], request_type: str
    ):
        try:
            await self.ensure_processing_loop_started()
            request_id = str(uuid4())
            self.response_futures[request_id] = asyncio.Future()
            await self.queue.put((request_data, request_type, request_id))
            return await self.response_futures[request_id]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal Server Error {e}")


app = FastAPI()

# Initialize the model and request processor
model = m3Wrapper("BAAI/bge-m3", device=device, compile=compile)
processor = RequestProcessor(
    model, accumulation_timeout=request_flush_timeout, max_request_to_flush=max_request
)


# Adding a middleware returning a 504 error if the request processing time is above a certain threshold
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        start_time = time.time()
        return await asyncio.wait_for(call_next(request), timeout=request_time_out)

    except asyncio.TimeoutError:
        process_time = time.time() - start_time
        return JSONResponse(
            {
                "detail": "Request processing time excedeed limit",
                "processing_time": process_time,
            },
            status_code=HTTP_504_GATEWAY_TIMEOUT,
        )


@app.post("/embeddings/", response_model=EmbedResponse)
async def get_embeddings(request: EmbedRequest):
    embeddings = await processor.process_request(request, "embed")
    return EmbedResponse(embeddings=embeddings)


@app.post("/rerank/", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    scores = await processor.process_request(request, "rerank")
    return RerankResponse(scores=scores)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port, workers=32, reload=False)
