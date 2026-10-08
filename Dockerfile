FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt
COPY inspect_model.py test_base.py lora_config.py ./
ENV HF_HOME=/hf-cache
CMD ["python", "inspect_model.py"]
