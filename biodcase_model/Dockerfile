FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY biodcase_edge ./biodcase_edge
COPY config ./config
COPY tests ./tests
COPY README.md TASKLIST.md ./

CMD ["python", "-m", "biodcase_edge.cli.train", "--config-name", "debug"]

