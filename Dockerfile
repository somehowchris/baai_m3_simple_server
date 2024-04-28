FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt /app

RUN apt update && apt install -y libpq-dev gcc

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Run m3_server.py when the container launches
CMD ["python", "m3_server.py"]