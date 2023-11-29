# build image
```bash
# 构建镜像命令
docker build -t registry.cn-hangzhou.aliyuncs.com/treasures/training-script-env:latest -f /training/python/training-script.Dockerfile .

docker run -it -d -v /root/workspace/YJX/auto-ml/volume/model:/treasures/model -v /root/workspace/YJX/auto-ml/volume/data:/treasures/data -v /root/workspace/YJX/auto-ml/volume/output:/treasures/output --name training-script registry.cn-hangzhou.aliyuncs.com/treasures/training-script-env:latest /bin/bash
```