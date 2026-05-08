FROM python:3.11-slim

WORKDIR /app

# Install CPU-only PyTorch first (avoids bundling 2 GB of CUDA libraries)
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    torchvision==0.17.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (grep strips torch lines already installed above)
COPY requirements.txt .
RUN grep -v "^torch" requirements.txt | pip install --no-cache-dir -r /dev/stdin

COPY . .

# Model weights are copied into app/model/ by deploy.sh before docker build
# Cloud Run expects port 8080
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
