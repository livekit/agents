FROM python:3.11

ARG AGENT

WORKDIR /usr/src/app

ADD ${AGENT}/ ${AGENT}/

WORKDIR "./${AGENT}"

RUN ls

RUN pip install -r requirements.txt

ENV AGENT_FILE=${AGENT}.py

ENTRYPOINT python $AGENT_FILE start