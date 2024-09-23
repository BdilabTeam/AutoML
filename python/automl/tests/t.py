def extract_metrics(data):
    # 获取 accuracy 和 loss 的最大长度队列
    accuracy = max(data['metrics/accuracy_top1'].values(), key=len)
    loss = max(data['val/loss'].values(), key=len)

    # 将 deque 对象转换为列表
    accuracy_list = list(accuracy)
    loss_list = list(loss)

    # 创建结果字典
    result = {
        'accuracy': accuracy_list,
        'loss': loss_list
    }

    return result

# 假设 'data' 是您提供的原始数据结构
data = {
    'metrics/accuracy_top1': {
        '5': deque([0.54545, 0.54545, 0.57576, 0.5757575631141663, 0.5757575631141663], maxlen=5),
        '10': deque([0.39394, 0.45455, 0.48485, 0.48485, 0.54545, 0.54545, 0.54545, 0.57576, 0.5757575631141663, 0.5757575631141663], maxlen=10)
    },
    'val/loss': {
        '5': deque([1.07973, 1.09624, 1.10444, 1.07515, 1.05233], maxlen=5),
        '10': deque([1.05169, 1.149, 1.08051, 1.08846, 1.06762, 1.07973, 1.09624, 1.10444, 1.07515, 1.05233], maxlen=10)
    },
    # ... 其他键值对 ...
}

# 调用函数并打印结果
result = extract_metrics(data)
print(result)
