from __future__ import absolute_import, division, print_function

import math, json, os, sys

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras as keras
# from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, SimpleRNN, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow_datasets as tfds
import argparse
from datetime import datetime
import pynvml
import yaml


nToMi = 1024 ** 2

# 导入数据大小

# os.environ['CUDA_VISIBLE_DEVICE'] = '0'
# 测GPU占用
# tf.config.experimental_run_functions_eagerly(True)
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

# pynvml.nvmlInit()
# handle = pynvml.nvmlDeviceGetHandleByIndex(0)


# @ tf.function
def main(args):

    BATCH_SIZE_PER_REPLICA = args.batch_size
    LEARNING_RATE = args.learning_rate
    EPOCHS = args.epochs
    DIR = args.data_dir
    workload_name = args.workload_name
    WORKLOAD_DIR = os.path.join(DIR, workload_name)

    DATA_DIR = os.path.join(DIR, 'Datasets')
    BUFFER_SIZE = 1000
    print('strategy start')
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        communication=tf.distribute.experimental.CollectiveCommunication.NCCL)
    print('strategy end')

    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    embedding = os.path.join(DATA_DIR, '2')
    hub_layer = hub.load(embedding)

    # def get_embedding(text, label):
    #     text = tf.cast(text, dtype=tf.string)
    #     return hub_layer(text), label

    def Input_fn():
        dataset, info = tfds.load(name='imdb_reviews', data_dir=DATA_DIR, with_info=True,
                                  as_supervised=True, download=False)
        train_dataset, test_dataset = dataset['train'], dataset['test']
        examples = []
        labels = []
        for example, label in train_dataset:
            examples.append(example.numpy())
            labels.append(label.numpy())
        examples = hub_layer(examples)
        examples = tf.expand_dims(examples, 1)
        train_dataset = tf.data.Dataset.from_tensor_slices((examples, labels))
        return train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE), len(train_dataset)

    if os.path.exists(WORKLOAD_DIR + '/worker-' + str(TASK_INDEX) + '.txt'):
        file = open(WORKLOAD_DIR + '/worker-' + str(TASK_INDEX) + '.txt', 'w').close()  # 清空文件


    # dataset, info = tfds.load('imdb_reviews', data_dir=DATA_DIR, with_info=True,
    #                           as_supervised=True, download=False)
    # train_dataset, test_dataset = dataset['train'], dataset['test']
    # train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    with strategy.scope():
        ds_train, length = Input_fn()
        print('length: ', length)
        # train_dataset = ds_train
        steps = math.floor(length / BATCH_SIZE) + 1


        # load data
        VOCAB_SIZE = 1000
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.DATA
        ds_train = ds_train.repeat()
        ds_train = ds_train.with_options(options)
        print('ds_train done')
        # encoder = TextVectorization(max_tokens=VOCAB_SIZE)
        # encoder.adapt(train_dataset.map(lambda text, label: text))
        # print('encoder.get_vocabulary: ', len(encoder.get_vocabulary()))

        model = Sequential([
            Input(shape=(1, 50, )),
            # encoder,
            # Embedding(
            #     input_dim=VOCAB_SIZE,
            #     output_dim=64,
            #     # Use masking to handle the variable sequence lengths
            #     mask_zero=True),
            # hub_layer,
            Bidirectional(LSTM(64, return_sequences=True)),
            # SimpleRNN(64),
            Dense(16, activation='relu'),
            Dense(1)
        ])

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      optimizer=Adam(LEARNING_RATE),
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
    model.fit(ds_train, steps_per_epoch=steps, epochs=EPOCHS,
                        callbacks=callbacks)
    print(model.summary())

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
    parser.add_argument('--workload_name', type=str, required=False, default='rnn')
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