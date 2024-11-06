# Model Server Framework
# Env Prepare:
```bash
# 激活虚拟环境
conda activate xxx / source ${VIRTUAL_ENV_PATH}/bin/activate
# 更新虚拟环境中的pip包
pip install --upgrade pip
# 在虚拟环境中安装poetry
pip install poetry
# 通过poetry进行依赖包安装
poetry install
```
# Start Spec:
```bash
cd tserve/python/tserve

python -m test

# 等待服务启动完成，终端执行以下命令进行服务调用测试:
curl -X POST -H "Content-Type: application/json" -d '{"instances": ["Wow!"]}' http://localhost:8080/v2/models/model/infer
```

