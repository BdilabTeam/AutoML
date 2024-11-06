# Image-Classification Server
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
cd tserve/python/image_classification_server

# 替换model_dir为真实的模型存储路径
python -m python -m image_classification_server --model_name=${model_name} --model_dir=${model_dir}

# 启动完成，服务调用测试:
curl -X POST -H "Content-Type: application/json" -d '{"instances": ["${data_file_path}"]}' http://localhost:8080/v2/models/${model_name}/infer
```

