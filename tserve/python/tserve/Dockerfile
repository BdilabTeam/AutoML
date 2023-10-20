FROM python:3.8.9

RUN mkdir treasures
WORKDIR /treasures

# mesa-libGL: this is to avoid "ImportError: libGL.so.1" from opencv
RUN apt-get update && apt-get install -y libgl1-mesa-glx

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple