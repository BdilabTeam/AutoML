from __future__ import absolute_import, division, print_function

import math, json, os, sys

import tensorflow as tf
import tensorflow.keras as keras
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers.experimental.preprocessing import Normalization, Resizing
from datetime import datetime
import argparse
import pathlib
import math
import numpy as np
import pynvml
import yaml

nToMi = 1024 ** 2

input_shape = (124, 129, 1)

# os.environ['CUDA_VISIBLE_DEVICE'] = '0'
# 测GPU占用
tf.config.experimental_run_functions_eagerly(True)
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)


# pynvml.nvmlInit()
# handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    # Note: You'll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph.
    return parts[-2]


def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


def get_spectrogram(waveform):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    return spectrogram


@tf.function
def main(args):
    BATCH_SIZE_PER_REPLICA = args.batch_size
    LEARNING_RATE = args.learning_rate
    EPOCHS = args.epochs
    DIR = args.data_dir
    workload_name = args.workload_name
    WORKLOAD_DIR = os.path.join(DIR, workload_name)

    DATA_DIR = os.path.join(DIR, 'Datasets')
    train_data_dir = pathlib.Path(os.path.join(DATA_DIR, 'mini_speech_commands'))
    commands = np.array(tf.io.gfile.listdir(str(train_data_dir)))
    commands = commands[commands != 'README.md']
    AUTOTUNE = tf.data.AUTOTUNE

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        communication=tf.distribute.experimental.CollectiveCommunication.NCCL)

    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    def get_spectrogram_and_label_id(audio, label):
        spectrogram = get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        label_id = tf.argmax(label == commands)
        return spectrogram, label_id

    def Input_fn():
        filenames = tf.io.gfile.glob(str(train_data_dir) + '/*/*')
        filenames = tf.random.shuffle(filenames)
        train_files = filenames[:6400]

        files_ds = tf.data.Dataset.from_tensor_slices(train_files)
        waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
        spectrogram_ds = waveform_ds.map(
            get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
        train_ds = spectrogram_ds
        length = len(train_ds)
        train_ds = train_ds.batch(BATCH_SIZE)
        train_ds = train_ds.cache().prefetch(AUTOTUNE).repeat()
        return train_ds, length, spectrogram_ds

    if os.path.exists(WORKLOAD_DIR + '/worker-' + str(TASK_INDEX) + '.txt'):
        file = open(WORKLOAD_DIR + '/worker-' + str(TASK_INDEX) + '.txt', 'w').close()  # 清空文件

    with strategy.scope():
        # load data
        ds_train, length, spectrogram_ds = Input_fn()
        steps = math.floor(length / BATCH_SIZE) + 1
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.DATA
        ds_train = ds_train.with_options(options)

        num_labels = len(commands)
        norm_layer = Normalization()
        norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))

        model = Sequential([
            Input(shape=input_shape),
            Resizing(32, 32),
            norm_layer,
            Conv2D(32, 3, activation='relu'),
            Conv2D(64, 3, activation='relu'),
            MaxPooling2D(),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_labels),
        ])

        model.compile(
            optimizer=Adam(LEARNING_RATE),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

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
    parser.add_argument('--workload_name', type=str, required=False, default='cnn-l')
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
