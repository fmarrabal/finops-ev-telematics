FROM python:3.11-slim

WORKDIR /app

COPY requirements-lock.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-lock.txt

COPY . .
RUN python -m pip install --no-deps -e .

CMD ["python", "reproduce.py"]
