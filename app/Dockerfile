FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y cron
# сертификат yandex cloud
RUN apt-get update && apt-get install -y wget && \ 
    mkdir -p ./certificate && \ 
    wget "https://storage.yandexcloud.net/cloud-certs/RootCA.pem" \ 
    --output-document ./certificate/RootCA.pem && \ 
    chmod 0655 ./certificate/RootCA.pem

COPY .env .
COPY reqs.txt .
RUN pip install -r reqs.txt

COPY crontask /etc/cron.d/crontask
RUN chmod 0644 /etc/cron.d/crontask && crontab /etc/cron.d/crontask
RUN mkdir -p /var/log/cronlogs

COPY app/main.py .


CMD cron -f