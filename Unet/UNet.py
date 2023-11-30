from __future__ import absolute_import, division, print_function

import math, json, os, sys

import tensorflow as tf
import argparse
from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense, Input, Concatenate, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import pynvml
import yaml

# workload_name = 'unet'

nToMi = 1024 ** 2


# os.environ['CUDA_VISIBLE_DEVICE'] = '0'
# 测GPU占用
tf.config.experimental_run_functions_eagerly(True)
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

# pynvml.nvmlInit()
# handle = pynvml.nvmlDeviceGetHandleByIndex(0)


def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask


def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask


def load_image_train(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask


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

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        communication=tf.distribute.experimental.CollectiveCommunication.NCCL)

    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    def Input_fn():
        BUFFER_SIZE = 1000
        dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True, data_dir=DATA_DIR, download=False, split='train[:{}%]'.format(int(PERCENTAGE * 100)))
        TRAIN_LENGTH = info.splits['train'].num_examples * PERCENTAGE
        train = dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return train_dataset, TRAIN_LENGTH

    if os.path.exists(WORKLOAD_DIR + '/worker-' + str(TASK_INDEX) + '.txt'):
        file = open(WORKLOAD_DIR + '/worker-' + str(TASK_INDEX) + '.txt', 'w').close()  # 清空文件

    with strategy.scope():
        # load data
        ds_train, TRAIN_LENGTH = Input_fn()
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.DATA
        ds_train = ds_train.with_options(options)

        STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

        base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=[128, 128, 3], include_top=False, weights=None)

        # 使用这些层的激活设置
        layer_names = [
            'block_1_expand_relu',  # 64x64
            'block_3_expand_relu',  # 32x32
            'block_6_expand_relu',  # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',  # 4x4
        ]
        layers = [base_model.get_layer(name).output for name in layer_names]

        # 创建特征提取模型
        down_stack = Model(inputs=base_model.input, outputs=layers)

        down_stack.trainable = False

        up_stack = [
            pix2pix.upsample(512, 3),  # 4x4 -> 8x8
            pix2pix.upsample(256, 3),  # 8x8 -> 16x16
            pix2pix.upsample(128, 3),  # 16x16 -> 32x32
            pix2pix.upsample(64, 3),  # 32x32 -> 64x64
        ]

        def unet_model():
            inputs = Input(shape=[128, 128, 3])
            x = inputs

            # 在模型中降频取样
            skips = down_stack(x)
            x = skips[-1]
            skips = reversed(skips[:-1])

            # 升频取样然后建立跳跃连接
            for up, skip in zip(up_stack, skips):
                x = up(x)
                concat = Concatenate()
                x = concat([x, skip])

            # 这是模型的最后一层
            last = Conv2DTranspose(
                3, 3, strides=2,
                padding='same')  # 64x64 -> 128x128

            x = last(x)

            return tf.keras.Model(inputs=inputs, outputs=x)

        model = unet_model()
        model.compile(optimizer=Adam(LEARNING_RATE),
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
    model.fit(ds_train, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,
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
    parser.add_argument('--data_percentage', type=float, required=False, default=1.0)
    parser.add_argument('--workload_name', type=str, required=False, default='unet')
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