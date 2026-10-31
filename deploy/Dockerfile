FROM python:3.11-slim

WORKDIR /app
COPY config/ ./config/
COPY data/ ./data/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY deploy/ ./deploy/

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["python3", "deploy/cloud_run_service.py"]
