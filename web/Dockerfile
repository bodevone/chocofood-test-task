# FROM python:3.8.3-alpine

# ENV PYTHONUNBUFFERED 1

# RUN mkdir /app

# WORKDIR /app

# RUN apk add --no-cache --update \
#     python3 python3-dev gcc \
#     gfortran musl-dev linux-headers
# # RUN apk --no-cache add musl-dev linux-headers g++

# RUN pip install --upgrade pip

# RUN pip install numpy
# RUN pip install scipy

# COPY requirements.txt /app/
# RUN pip install -r requirements.txt

# COPY . /app/

FROM ubuntu:latest

RUN apt-get update -y

RUN apt-get install -y python3-pip python3-dev build-essential

COPY . /app

WORKDIR /app

RUN pip3 install -r requirements.txt