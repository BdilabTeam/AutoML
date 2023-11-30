FROM tensorflow/tensorflow:2.6.0-gpu
RUN mkdir /train/Datasets
COPY /home/bdilab/Datasets /train/Datasets
ADD . .
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -y && apt-get install tzdata \
    && ln -snf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata
ENV TZ Asia/Shanghai
RUN pip install -i https://pypi.douban.com/simple/ -r requirements.txt
