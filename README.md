# AutoML
##自动化机器学习训练任务调度模块
## TFJob目录
TFJob目录中有8个TFJob（k8s中的资源类型）的配置文件，代表任务调度实验所选取的8类机器学习任务在k8s环境中部署时的配置文件。目录中的8个yaml文件是模板，真实实验时根据训练任务不同的batch_size，epochs，workload，在代码运行时依据Yam模板为每个作业创建对应的yaml文件。
