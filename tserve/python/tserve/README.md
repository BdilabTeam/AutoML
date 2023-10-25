# Model Server Framework
# Start Spec:
```
cd tserve/python/tserve

python -m test

服务启动完成，服务调用测试:
curl -X POST -H "Content-Type: application/json" -d '{"instances": ["Wow!"]}' http://localhost:8080/v2/models/model/infer"
```

