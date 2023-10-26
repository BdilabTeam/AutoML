# 模型训练应用
# Env Prepare:
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
# Start training_app
```bash
cd training/python/training_app/training_app

python training_controller.py
```