FROM python:3.11-slim

WORKDIR /app

COPY policy-llm-ops/pyproject.toml ./
RUN pip install --no-cache-dir .

COPY policy-llm-ops/llm_ops ./llm_ops
COPY contracts /contracts

ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "llm_ops.gateway:app", "--host", "0.0.0.0", "--port", "8000"]
