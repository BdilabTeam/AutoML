FROM tensorflow/tensorflow:2.6.0-gpu
ADD . .
ENV DEBIAN_FRONTEND noninteractive
RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-get update -y && apt-get install tzdata \
    && ln -snf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata
ENV TZ Asia/Shanghai
RUN pip install -i https://pypi.douban.com/simple/ -r requirements.txt