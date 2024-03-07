# Project Structure Spec
```bash
'automl' project is composed of the 'alserver, autoselect, autotrain, autoschedule'.
* 'alserver' is geared towards web applications
* 'autoselect' is used to automate model selection
* 'autotrain' is used to automate model training
* 'autoschedule' is used to automate the scheduling of training 
```

# Env Prepare:
```bash
# 激活虚拟环境
conda activate xxx or source ${VIRTUAL_ENV_PATH}/bin/activate
# 更新虚拟环境中的pip包
pip install --upgrade pip
# 在虚拟环境中安装poetry
pip install poetry
# 在虚拟环境中安装poetry, 按需安装可选依赖组
poetry install --with autoselect,autoschedule,storage,autotrain,test --no-root --no-cache
# 创建数据库/表
CREATE DATABASE automl CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
pytest -s -k test_generate_schemas test_mysql_server.py
```

# Start Spec:
```bash
cd .

python -m alserver # 等待服务启动完成，进入swagger页面：http://localhost:8000/docs
```

# Test Spec
```bash
# 执行'pytest'命令，运行所有测试脚本
pytest
# 运行'某个'测试脚本
pytest {script_name}
# '-s'参数，输出print日志
pytest -s {script_name}
# 执行指定py文件中的指定测试函数
pytest -k {test_func} {test_file.py}
```