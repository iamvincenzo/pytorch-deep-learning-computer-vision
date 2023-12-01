# Use a base Ubuntu image
FROM ubuntu:20.04

# set noninteractive mode and timezone
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Rome

# Install basic dependencies
RUN apt-get update && \
    apt-get install -y \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU version
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip3 install pytorch-lightning tensorboardX tensorboard rich

# Set the working directory inside the container
WORKDIR /app

# Copy your application code into the container
COPY ./main.py .

# Set the command to run your application
CMD ["python3", "main.py"]
