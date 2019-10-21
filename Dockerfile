FROM tensorflow/tensorflow:latest-gpu-py3 as tf-base
WORKDIR /usr/local/tf-demo
COPY requirements-docker.txt ./
RUN pip3 install --upgrade pip && pip3 install -r requirements-docker.txt

FROM tf-base
WORKDIR /usr/local/tf-demo
COPY ./src/ ./src/
CMD ["python", "./src/tf-demo.py"]
