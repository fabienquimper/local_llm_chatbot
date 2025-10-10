# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies for pypdf, sentence-transformers, torch CPU build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# Use CPU-only PyTorch by default; users with GPUs should prefer host setup
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Default PDF dir
RUN mkdir -p /app/pdf

EXPOSE 7860

CMD ["python", "main_rag_chat.py", "--timeout", "240", "--max-tokens", "512"]


