# AutoML is a module for Web applications

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
cd /AutoML/python/automl

python -m automl

# 等待服务启动完成，进入swagger页面：http://localhost:8000/docs
```

# Test Spec
```bash
# 执行'pytest'命令，运行所有测试脚本
pytest

# 运行'某个'测试脚本
pytest {script_name}

# '-s'参数，输出print日志
pytest -s {script_name}
```

# Project Structure Spec
```bash
.
├── README.md                   (项目环境说明)
├── pyproject.toml              (基于poetry的项目依赖管理)
├── automl                      (项目代码)
    ├── handlers                (业务处理包)
        ├── dataplane.py        (定义具体业务处理逻辑, 类比service层)
    ├── rest                    (REST服务包)
        ├── app.py              (Fastapi应用程序)
        ├── endpoints.py        (REST端点的实现, 类比controller层)
        ├── errors.py           (定义Fastapi应用程序全局异常捕获)
        ├── logging.py          (封装logging模块, 用于统一日志管理)
        ├── requests.py         (封装Fastapi的Request, 用于统一请求处理，用于编解码, 暂时用不到)
        ├── responses.py        (封装Fastapi的Response, 用于统一相应处理，用于编解码, 暂时用不到)
        ├── server.py           (封装Uvicorn, 主要负责启动、停止Uvicorn服务器)
    ├── schemas                 (数据模型包)
        ├── input_schemas.py    (请求输入数据模型)
        ├── output_schemas.py   (响应输出数据模型)
    ├── utils                   (项目工具包)
        ├── logging.py          (日志工具模块)
    ├── errors.py               (自定义异常模块)
    ├── settings.py             (项目配置入口点)
    ├── server.py               ()
    ├── version.py              (项目迭代版本)
    ├── __main__.py             (项目启动入口点)
```

