from __future__ import absolute_import, division, print_function

import math, json, os, sys
import pathlib
import argparse
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from model import *
from datetime import datetime
import tensorflow_text as tf_text
import pynvml
import yaml



tf.config.experimental_run_functions_eagerly(True)
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)


# 加载数据，英语-西班牙语
def load_data(path, PERCENTAGE):
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines()
    pairs = [line.split('\t') for line in lines]
    inp = [inp for targ, inp in pairs]
    targ = [targ for targ, inp in pairs]
    length = math.floor(len(targ) * PERCENTAGE) + 1
    return targ[:length], inp[:length]


# 文本标准化
def tf_lower_and_split_punct(text):
    # Split accecented characters.
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    # Strip whitespace.
    text = tf.strings.strip(text)

    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    # [START] ¿ todavia esta en casa ? [END]
    return text

@ tf.function
def main(args):
    BATCH_SIZE_PER_REPLICA = args.batch_size
    LEARNING_RATE = args.learning_rate
    EPOCHS = args.epochs
    DIR = args.data_dir
    PERCENTAGE = args.data_percentage
    workload_name = args.workload_name
    WORKLOAD_DIR = os.path.join(DIR, workload_name)
    DATA_DIR = os.path.join(DIR, 'Datasets')
    path_to_file = pathlib.Path(DATA_DIR)/'spa-eng/spa.txt'
    embedding_dim = 64
    units = 128
    max_vocab_size = 5000
    BUFFER_SIZE = 1000


    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        communication=tf.distribute.experimental.CollectiveCommunication.NCCL)

    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    def Input_fn():
        targ, inp = load_data(path_to_file, PERCENTAGE)
        length = len(targ)
        dataset = tf.data.Dataset.from_tensor_slices((inp, inp)).shuffle(BUFFER_SIZE)
        dataset = dataset.batch(BATCH_SIZE)
        return dataset, targ, inp, length

    # if not os.path.exists(WORKLOAD_DIR):
    #     os.makedirs(WORKLOAD_DIR)

    if os.path.exists(WORKLOAD_DIR + '/worker-' + str(TASK_INDEX) + '.txt'):
        file = open(WORKLOAD_DIR + '/worker-' + str(TASK_INDEX) + '.txt', 'w').close()  # 清空文件

    with strategy.scope():
        # load data
        ds_train, targ, inp, length = Input_fn()
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.DATA
        ds_train = ds_train.repeat()
        ds_train = ds_train.with_options(options)
        num_train_steps = math.floor(length / BATCH_SIZE) + 1

        input_text_processor = preprocessing.TextVectorization(standardize=tf_lower_and_split_punct,
                                                               max_tokens=max_vocab_size)
        input_text_processor.adapt(inp)

        output_text_processor = preprocessing.TextVectorization(standardize=tf_lower_and_split_punct,
                                                                max_tokens=max_vocab_size)
        output_text_processor.adapt(targ)

        translator = TrainTranslator(embedding_dim, units, input_text_processor=input_text_processor, output_text_processor=output_text_processor)
        translator.compile(optimizer=tf.optimizers.Adam(LEARNING_RATE), loss=MaskedLoss())


    class PrintLR(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):  # pylint: disable=no-self-use
            # epoch, time, GPU
            # print('\nLearning rate for epoch {} is {}'.format(
            #     epoch + 1, finetuned_model.optimizer.lr.numpy())
            with open(WORKLOAD_DIR + '/worker-' + str(TASK_INDEX) + '.txt', 'a+') as f:
                f.write('{}, {}\n'.format(epoch, datetime.now()))

    callbacks = [
        # tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        # tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
        #                                    save_weights_only=True),
        PrintLR()
    ]

    # model fit
    translator.fit(ds_train, steps_per_epoch=num_train_steps, epochs=EPOCHS, callbacks=callbacks)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=False, default=16)
    parser.add_argument('--learning_rate', type=float, required=False, default=0.0001)
    parser.add_argument('--epochs', type=int, required=False, default=100)
    parser.add_argument('--data_percentage', type=float, required=False, default=0.2)
    parser.add_argument('--workload_name', type=str, required=False, default='seq2seq')
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

    args = parse_arguments()
    main(args)