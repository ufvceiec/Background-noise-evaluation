FROM tensorflow/tensorflow:latest-gpu

WORKDIR /home/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN apt install nano -y

