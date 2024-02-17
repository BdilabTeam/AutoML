import os
import pandas as pd
import random

# 生成随机数据的函数
def generate_random_data(num_samples):
    data = []
    for _ in range(num_samples):
        # 随机生成钢材成分数据
        steel_composition = {element: round(random.uniform(0, 1), 2) for element in ['C', 'Si', 'Mn', 'P', 'S', 'Ni', 'Mo', 'Cr', 'W', 'Cu']}

        # 随机生成热轧参数和淬火参数
        hot_rolling_params = {
            '变形温度T': random.randint(700, 1000),
            '变形程度': random.choice(['0.3', '0.4', '0.5', '0.6']),    # 0.3: 30%
            '变形速率': round(random.uniform(3, 7), 2)  # 单位 m/s
        }

        quenching_params = {
            '淬火温度': random.randint(800, 1100)
        }

        # 生成淬透性目标值, 1:优, 0: 良, -1: 一般
        quenching_quality = random.choice(['1', '0', '-1'])
        # quenching_quality = random.choice(['0.11', '1.0'])

        # 合并生成的数据
        row_data = {**steel_composition, **hot_rolling_params, **quenching_params, '淬透性': quenching_quality}
        data.append(row_data)

    return data

# 生成10万条数据
num_samples = 500
data = generate_random_data(num_samples)

# 转换为DataFrame
df = pd.DataFrame(data)

# 将数据保存为CSV文件
df.to_csv(os.path.join(os.path.dirname(__file__), 'structured-data-classification.csv'), index=False)