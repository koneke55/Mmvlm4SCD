FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY environment/requirements.txt /workspace/requirements.txt
RUN pip install -U pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install -r /workspace/requirements.txt

COPY . /workspace
RUN pip install -e .

CMD ["python", "src/scripts/run_full_experiment.py", "--config", "configs/default.yaml"]
