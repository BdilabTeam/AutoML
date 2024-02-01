# 'autoselect' is the core module of 'automl'

# Env Prepare:
```bash
# 激活虚拟环境
conda activate xxx / source ${VIRTUAL_ENV_PATH}/bin/activate
# 更新虚拟环境中的pip包
pip install --upgrade pip
# 在虚拟环境中安装poetry
pip install poetry
# 通过poetry进行依赖包安装
poetry install --with test --no-root --no-cache
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
