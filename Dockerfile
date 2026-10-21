
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --no-cache-dir --default-timeout=600 --retries=10 -r requirements.txt

COPY app.py .
COPY shopping_model.pkl .

EXPOSE 5000

CMD ["python", "app.py"]