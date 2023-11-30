from __future__ import absolute_import, division, print_function

import math, json, os, sys
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import argparse
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import tensorflow_datasets as tfds
from model import *
import pynvml
import yaml
from tqdm import tqdm

# workload_name = 'resnet'

nToMi = 1024 ** 2

# 导入数据大小
noise_dim = 100

# os.environ['CUDA_VISIBLE_DEVICE'] = '0'
# 测GPU占用
tf.config.experimental_run_functions_eagerly(True)
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

# pynvml.nvmlInit()
# handle = pynvml.nvmlDeviceGetHandleByIndex(0)


def Input_fn():
    BUFFER_SIZE = 10000
    dataset = tfds.load(name='mnist', data_dir=DATA_DIR, download=False, split='train[:{}%]'.format(int(PERCENTAGE * 100)))
    train_dataset = dataset
    examples = []
    for ds in train_dataset:
        examples.append(ds['image'].numpy())
    train_images = tf.convert_to_tensor(examples, dtype=tf.float32)
    # train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return train_dataset


def on_epoch_end(epoch):
    with open(WORKLOAD_DIR + '/worker-' + str(TASK_INDEX) + '.txt', 'a+') as f:
        f.write('{}, {}\n'.format(epoch, datetime.now()))


@tf.function
def step_fn(inputs):
    noise = tf.random.normal([inputs.shape[0], noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(inputs, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss


@tf.function
def train_step(images):
    gen_loss, disc_loss = strategy.run(step_fn, args=(images,))
    # gen_mean_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, gen_loss, axis=0)
    # disc_mean_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, disc_loss, axis=0)
    return tf.reduce_mean(gen_loss), tf.reduce_mean(disc_loss)


def train(dataset, epochs):
    with strategy.scope():
        for epoch in range(epochs):
            for image_batch in tqdm(dataset):
                gen_loss, disc_loss = train_step(image_batch)
            on_epoch_end(epoch)
            print(epoch, gen_loss.numpy(), disc_loss.numpy())
            # if (epoch + 1) % 15 == 0:
            #     checkpoint.save(file_prefix=checkpoint_prefix)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=False, default=16)
    parser.add_argument('--learning_rate', type=float, required=False, default=0.0001)
    parser.add_argument('--epochs', type=int, required=False, default=100)
    parser.add_argument('--data_percentage', type=float, required=False, default=0.5)
    parser.add_argument('--workload_name', type=str, required=False, default='dcgan')
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
    with strategy.scope():
        # load data
        ds_train = Input_fn()
        # options = tf.data.Options()
        # options.experimental_distribute.auto_shard_policy = \
        #     tf.data.experimental.AutoShardPolicy.DATA
        # ds_train = ds_train.with_options(options)

        generator = build_generator()
        discriminator = build_discriminator()
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

        def discriminator_loss(real_output, fake_output):
            real_loss = cross_entropy(tf.ones_like(real_output), real_output)
            fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
            total_loss = real_loss + fake_loss
            return total_loss

        def generator_loss(fake_output):
            return cross_entropy(tf.ones_like(fake_output), fake_output)

        generator_optimizer = Adam(LEARNING_RATE)
        discriminator_optimizer = Adam(LEARNING_RATE)
    if os.path.exists(WORKLOAD_DIR + '/worker-' + str(TASK_INDEX) + '.txt'):
        file = open(WORKLOAD_DIR + '/worker-' + str(TASK_INDEX) + '.txt', 'w').close()  # 清空文件

    # checkpoint_dir = os.path.join(WORKLOAD_DIR, 'checkpoints')
    # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    # checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
    #                                  discriminator_optimizer=discriminator_optimizer,
    #                                  generator=generator,
    #                                  discriminator=discriminator)
    # model train
    train(ds_train, EPOCHS)
