FROM python:3.12-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('jinaai/jina-embeddings-v3', trust_remote_code=True); AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v3', trust_remote_code=True)"

FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

COPY . .

ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

HEALTHCHECK CMD curl --fail http://localhost:5000/health || exit 1

CMD ["python", "app.py"]