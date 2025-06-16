FROM python:3.12-slim

WORKDIR /app

COPY reqs.api.txt reqs.txt
RUN pip install -r reqs.txt

COPY app/api/main.py .

