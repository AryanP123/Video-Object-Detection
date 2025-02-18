FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir \
    flask \
    ultralytics \
    torch \
    opencv-python-headless

EXPOSE 5000

CMD ["python", "app.py"]