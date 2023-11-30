from __future__ import absolute_import, division, print_function

import math, json, os, sys

import tensorflow as tf
# import tensorflow.keras as keras
import argparse
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow_datasets as tfds

from datetime import datetime
import pynvml
import yaml

# workload_name = 'cnn'

nToMi = 1024 ** 2

# 导入数据大小
# SIZE = (224, 224)

# os.environ['CUDA_VISIBLE_DEVICE'] = '0'
# 测GPU占用
tf.config.experimental_run_functions_eagerly(True)
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

# pynvml.nvmlInit()
# handle = pynvml.nvmlDeviceGetHandleByIndex(0)


@ tf.function
def main(args):

    BATCH_SIZE_PER_REPLICA = args.batch_size
    LEARNING_RATE = args.learning_rate
    EPOCHS = args.epochs
    DIR = args.data_dir
    workload_name = args.workload_name
    WORKLOAD_DIR = os.path.join(DIR, workload_name)

    DATA_DIR = os.path.join(DIR, 'Datasets')

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        communication=tf.distribute.experimental.CollectiveCommunication.NCCL)

    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    def Input_fn():
        BUFFER_SIZE = 1000
        dataset, info = tfds.load(name='cifar10', data_dir=DATA_DIR, with_info=True, download=False)
        train_dataset, test_dataset = dataset['train'], dataset['test']
        examples = []
        labels = []
        for ds in train_dataset:
            examples.append(ds['image'].numpy())
            labels.append(ds['label'].numpy())
        train_dataset = tf.data.Dataset.from_tensor_slices((examples, labels))
        train_dataset = train_dataset.shuffle(buffer_size=BUFFER_SIZE)
        return train_dataset

    num_train_steps = math.floor(50000/BATCH_SIZE)+1

    if os.path.exists(WORKLOAD_DIR + '/worker-' + str(TASK_INDEX) + '.txt'):
        file = open(WORKLOAD_DIR + '/worker-' + str(TASK_INDEX) + '.txt', 'w').close()  # 清空文件

    with strategy.scope():
        # load data
        ds_train = Input_fn().batch(BATCH_SIZE).repeat()
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.DATA
        ds_train = ds_train.with_options(options)

        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(10))

        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

    class PrintLR(tf.keras.callbacks.Callback):

        def on_epoch_end(self, epoch, logs=None):  # pylint: disable=no-self-use
            # epoch, time, GPU
            # print('\nLearning rate for epoch {} is {}'.format(
            #     epoch + 1, finetuned_model.optimizer.lr.numpy())
            with open(WORKLOAD_DIR + '/worker-' + str(TASK_INDEX) + '.txt', 'a+') as f:
                f.write('{}, {}\n'.format(epoch, datetime.now()))


    # checkpoints
    # checkpoint_dir = os.path.join(WORKLOAD_DIR, 'checkpoints')
    # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    callbacks = [
        # tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        # tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
        #                                    save_weights_only=True),
        PrintLR()
    ]

    # model fit
    model.fit(ds_train, steps_per_epoch=num_train_steps, epochs=EPOCHS,
                        callbacks=callbacks)

    # save model
    save_model_dir = os.path.join(WORKLOAD_DIR, 'model_path')
    def is_chief():
        return TASK_INDEX == 0

    if is_chief():
        model_path = save_model_dir

    else:
        # Save to a path that is unique across workers.
        model_path = save_model_dir + '/worker_tmp_' + str(TASK_INDEX)

    model.save(model_path)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=False, default=16)
    parser.add_argument('--learning_rate', type=float, required=False, default=0.0001)
    parser.add_argument('--epochs', type=int, required=False, default=100)
    parser.add_argument('--workload_name', type=str, required=False, default='cnn')
    parser.add_argument('--data_dir', type=str, required=False, default='/train')
    args = parser.parse_args()
    return args


# def get_GPU_limits():
#   with open('cnn.yaml', 'r') as spec_file:
#     spec_string = spec_file.read()
#     total_paras = yaml.load(spec_string, Loader=yaml.FullLoader)
#   GPU_limits = total_paras['spec']['tfReplicaSpecs']['Worker']['template']['spec']['containers'][0]['resources']['limits']['aliyun.com/gpu-mem']
#   return int(GPU_limits) * 1024 * 0.7 / (meminfo.total/nToMi)


if __name__ == "__main__":
    ## 限制GPU使用
    # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    # GPU_limits = get_GPU_limits()
    # tf.config.experimental_run_functions_eagerly(True)
    # gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=GPU_limits)
    # sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

    os.environ['NCCL_DEBUG'] = 'INFO'
    # to decide if a worker is chief, get TASK_INDEX in Cluster info
    tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
    TASK_INDEX = tf_config['task']['index']
    # TASK_INDEX = 0
    args = parse_arguments()
    main(args)