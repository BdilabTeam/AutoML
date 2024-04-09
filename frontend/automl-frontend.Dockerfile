# # 基于官方 Nginx 镜像
FROM nginx:alpine

# 拷贝自定义的 Nginx 配置
COPY nginx.conf /etc/nginx/nginx.conf

# 将 dist、build 文件夹中的文件复制到 Nginx 的静态资源目录下
RUN mkdir -p /usr/share/automl/web
COPY web /usr/share/automl/web

# 暴漏端口8888
EXPOSE 8080

# 运行nginx
ENTRYPOINT ["nginx", "-g", "daemon off;"]
