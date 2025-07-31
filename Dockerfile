# FROM python:3.11-slim-buster
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./
RUN uv pip install --system --no-deps .
COPY . .
COPY tfidf_vectorizer.pkl ./
RUN uv pip install --system .

CMD ["uv", "run", "python", "flask_app/app.py"]