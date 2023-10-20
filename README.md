# 自动化机器学习平台
    * 特征自动选择技术
    * 自动化模型选择技术
    * 超参数优化技术
    * 模型训练与部署技术
    * 模型训练任务调度技术
# 其余部分参考以下项目结构, 方便后续集成与维护。以YJX分支为例：
```
- project_name  （tserve）
    - docs
    - python
        - module_name   （tserve）
            - module1   （test）
                - **
            - module2   （tserve）
                - **
            - module_operation_maintenance_files    (README.md、Dockerfie、requirements.txt、...)
```
# 各小组提交代码至对应分支（例如LJY）, 主分支用于合并代码