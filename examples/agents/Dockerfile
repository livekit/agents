FROM python:3.11

ARG AGENT

WORKDIR /usr/src/app

ADD ${AGENT}/ ${AGENT}/

WORKDIR "./${AGENT}"

RUN ls

RUN apt-get -y update
RUN apt-get install -y ffmpeg

RUN pip install -r requirements.txt

ENV AGENT_FILE=${AGENT}.py

ENTRYPOINT python $AGENT_FILE start