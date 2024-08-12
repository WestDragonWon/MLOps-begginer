FROM python:3.11-buster 

COPY ./requirements.txt /tmp

RUN pip install -r /tmp/requirements.txt

COPY ./src/dataset /opt/stc/dataset
COPY src /opt/src
COPY start-api-server.sh /opt/src
WORKDIR /opt