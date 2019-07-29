FROM tensorflow/tensorflow:latest-py3
WORKDIR /usr/local/tf-demo
COPY ./src/ ./src/
CMD ["python", "./src/tf-demo.py"]
