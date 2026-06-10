FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY live_tools/ ./live_tools/

RUN mkdir -p logs

CMD ["python", "live_tools/run.py"]
