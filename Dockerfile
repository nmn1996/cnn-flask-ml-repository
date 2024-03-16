# Use a Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all the necessary files from the local directory to the working directory inside the container

COPY cnn-model.py serving.py model-run.sh ./

# Install any dependencies
RUN pip install flask numpy torchvision torch

# Run the model script
ENTRYPOINT ["sh", "model-run.sh"]

