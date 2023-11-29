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
# cd training/python/training_app/training_app  本机
cd /root/workspace/YJX/Env/yjx/app_new #服务器60.204.186.96

python training_controller.py
```



## 运行流程

### 1.创建项目

传递4个参数：name,task_type,is_automatic(false),model_name_or_path,data_name_or_path.
其实后两个参数应该是自动生成的不用传，因为训练测试的镜像在master上而node1上运行的这个fastapi项目路径和master有些不一样（app_new），所以手动传

![image-20231129105209361](C:\Users\86136\AppData\Roaming\Typora\typora-user-images\image-20231129105209361.png)



### 2.修改项目

不说了，后续考虑有的项目模型和数据已经传上去了还能改吗？
![image-20231129105334854](C:\Users\86136\AppData\Roaming\Typora\typora-user-images\image-20231129105334854.png)



### 3.开启训练

传一个id直接跑

![image-20231129105425897](C:\Users\86136\AppData\Roaming\Typora\typora-user-images\image-20231129105425897.png)



### 4.删除项目

考虑删除的时候要不要把数据和模型都删掉



### 5.在master上查看训练pod和日志

# 代码结构

## 是一个fastapi框架的项目，用来管理训练项目，开启训练并保存输出。

![image-20231128232509317](C:\Users\86136\AppData\Roaming\Typora\typora-user-images\image-20231128232509317.png)


### training_controller 接口类

对训练项目crud操作



### training_operator_client 

用于构建tfjob资源进行训练



### utils包

![image-20231128233630711](C:\Users\86136\AppData\Roaming\Typora\typora-user-images\image-20231128233630711.png)



