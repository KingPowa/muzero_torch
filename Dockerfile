# For more information, please refer to https://aka.ms/vscode-docker-python
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

ENV UID=1000
ENV GID=1001

RUN apt update -y && apt install -y python3 python3-pip git

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
RUN mkdir /app
COPY requirements.txt /app
RUN pip install -r /app/requirements.txt
RUN rm /app/requirements.txt

WORKDIR /app/muzero_torch

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

