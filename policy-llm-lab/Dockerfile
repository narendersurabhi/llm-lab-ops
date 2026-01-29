FROM python:3.11-slim

WORKDIR /app

COPY policy-llm-lab/pyproject.toml ./
RUN pip install --no-cache-dir .

COPY policy-llm-lab/llm_lab ./llm_lab
COPY policy-llm-lab/data ./data
COPY policy-llm-lab/artifacts ./artifacts
COPY contracts /contracts

ENV PYTHONUNBUFFERED=1

EXPOSE 8001
CMD ["uvicorn", "llm_lab.retrieval_api:app", "--host", "0.0.0.0", "--port", "8001"]
