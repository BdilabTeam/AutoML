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

# 代码结构

## 是一个fastapi框架的项目，用来管理训练项目，开启训练并保存输出。

![image-20231128232509317](C:\Users\86136\AppData\Roaming\Typora\typora-user-images\image-20231128232509317.png)


### training_controller 接口类

对训练项目crud操作



### training_operator_client 

用于构建tfjob资源进行训练



### utils包

![image-20231128233630711](C:\Users\86136\AppData\Roaming\Typora\typora-user-images\image-20231128233630711.png)



