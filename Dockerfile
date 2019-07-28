FROM tensorflow/tensorflow:latest-py3
WORKDIR /usr/local/tf-demo
COPY ./src/demo-verify-tf-install/main.py ./main.py
CMD ["python", "main.py"]
