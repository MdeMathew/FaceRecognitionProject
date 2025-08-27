FROM ubuntu:latest
LABEL authors="mathi"

ENTRYPOINT ["top", "-b"]