FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false EMBEDDING_OFFLINE=true FASTEMBED_CACHE_PATH=/opt/models
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# Bundle ONNX model at build time so startup never downloads weights.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN EMBEDDING_OFFLINE=false python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='${EMBEDDING_MODEL}',cache_dir='/opt/models',threads=1)"
COPY . .
RUN test -f data/index/manifest.json || (echo 'Missing index: run knowledge sync workflow first.' && exit 1)
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
