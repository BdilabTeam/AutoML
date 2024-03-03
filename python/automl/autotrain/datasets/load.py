import os
import glob
import tensorflow as tf
import numpy as np
from typing import Optional, Union, Sequence, Mapping

def load_dataset(
    path: str,
    name: Optional[str],
    data_dir: Optional[str],
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]]
):
    pass

def regression_image_dataset_from_directory(directory: str):
    # 数据准备
    train_dir = directory
    items = os.listdir(train_dir)
    # 获取'文件夹'名称
    folder_names = [item for item in items if os.path.isdir(os.path.join(train_dir, item))]

    file_paths = []
    labels = []
    for folder_name in folder_names:
        files = glob.glob(os.path.join(train_dir, folder_name, '*'))
        file_paths.extend(files)
        labels.extend([float(folder_name)] * len(files))
        
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    def load_image(file_path, label):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_image(image, channels=3)
        return image, label

    dataset = dataset.map(load_image)

    x_train = []
    y_train = []
    for x, y in dataset:
        x_train.append(x)
        y_train.append(y)
    # 将特征和标签转换为张量
    x_train = np.asarray(tf.stack(x_train))
    y_train = np.asarray(tf.stack(y_train))