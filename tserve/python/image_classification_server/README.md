# Image-Classification Server
# Start Spec:
```
cd tserve/python/image_classification_server

替换model_dir为真实的模型存储路径
python -m python -m image_classification_server --model_name=ic --model_dir={model_dir}

启动完成，服务调用测试:
curl -X POST -H "Content-Type: application/json" -d '{"instances": ["Wow!"]}' http://localhost:8080/v2/models/ic/infer"
```