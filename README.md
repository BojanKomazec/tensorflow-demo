# tensorflow-demo
Tensorflow demo project.

# Running in Docker container

To run this application in its Docker container it is first necessary to create its Docker image:
```
$ docker build -t tf-demo .
```
To launch td-demo container:
```
$ docker run -it --rm --name tf-demo tf-demo
```
