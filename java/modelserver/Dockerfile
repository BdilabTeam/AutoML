FROM openjdk:8-jdk-alpine

# 创建存放k8s config的文件夹
RUN mkdir -p /root/.kube

# 将config文件存入 /root/.kube 目录下
ADD config /root/.kube

# 在容器内创建一个目录来保存我们的 Spring Boot 应用
WORKDIR /app

# 将 Spring Boot 的 JAR 文件复制到容器内
ADD deployment-1.0-SNAPSHOT.jar /app

# 指定容器启动时应运行的命令
ENTRYPOINT ["java", "-jar", "/app/deployment-1.0-SNAPSHOT.jar"]
