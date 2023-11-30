from __future__ import absolute_import, division, print_function

import math, json, os, sys

import tensorflow as tf
import tensorflow.keras as keras
import argparse
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import pynvml
import yaml

# workload_name = 'vgg'

nToMi = 1024 ** 2

# 导入数据大小
SIZE = 64

# os.environ['CUDA_VISIBLE_DEVICE'] = '0'
# 测GPU占用
tf.config.experimental_run_functions_eagerly(True)
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

# pynvml.nvmlInit()
# handle = pynvml.nvmlDeviceGetHandleByIndex(0)


def load_sample(sample_dir):
    # 图片名列表
    lfilenames = []
    # 标签名列表
    labelnames = []
    # 遍历文件夹
    for (dirpath, dirnames, filenames) in os.walk(sample_dir):
        # 遍历图片
        for filename in filenames:
            # 每张图片的路径名
            filename_path = os.sep.join([dirpath, filename])
            # 添加文件名
            lfilenames.append(filename_path)
            # 添加文件名对应的标签
            labelnames.append(dirpath.split('/')[-1])

    # 生成标签名列表
    lab = list(sorted(set(labelnames)))
    # 生成标签字典
    labdict = dict(zip(lab, list(range(len(lab)))))
    # 生成与图片对应的标签列表
    labels = [labdict[i] for i in labelnames]
    # 图片与标签字典
    image_label_dict = dict(zip(lfilenames, labels))
    # 将文件名与标签列表打乱
    lfilenames = []
    labels = []
    for key in image_label_dict:
        lfilenames.append(key)
        labels.append(image_label_dict[key])
    # 返回文件名与标签列表
    return lfilenames, labels


@ tf.function
def main(args):

    BATCH_SIZE_PER_REPLICA = args.batch_size
    LEARNING_RATE = args.learning_rate
    EPOCHS = args.epochs
    DIR = args.data_dir
    workload_name = args.workload_name
    WORKLOAD_DIR = os.path.join(DIR, workload_name)

    DATA_DIR = os.path.join(DIR, 'Datasets')
    TRAIN_DIR = os.path.join(DATA_DIR, 'ImageNet')
    Classes = list(os.listdir(TRAIN_DIR))
    classes_num = len(Classes)

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        communication=tf.distribute.experimental.CollectiveCommunication.NCCL)

    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    def _parseone(filename, label):
        # 读取所有图片
        image_string = tf.io.read_file(filename)
        # 将图片解码并返回空的shape
        image_decoded = tf.image.decode_image(image_string, channels=3)
        # 因为是空的shape，所以需要设置shape
        image_decoded.set_shape([None, None, None])
        image_decoded = tf.image.resize(image_decoded, (SIZE, SIZE))
        # 归一化
        # image_decoded = image_decoded/255.
        # 将归一化后的像素矩阵转化为image张量
        image_decoded = tf.cast(image_decoded, dtype=tf.float32)
        # 将label转为张量
        label = tf.cast(tf.reshape(label, []), dtype=tf.int32)
        # 将标签制成one_hot
        label = tf.one_hot(label, depth=classes_num, on_value=1)
        return image_decoded, label

    def Input_fn():
        BUFFER_SIZE = 1000
        filenames, labels = load_sample(TRAIN_DIR)
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        return dataset.map(_parseone).cache().shuffle(BUFFER_SIZE)

    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
    num_train_steps = math.floor(num_train_samples / BATCH_SIZE) + 1

    if os.path.exists(WORKLOAD_DIR + '/worker-' + str(TASK_INDEX) + '.txt'):
        file = open(WORKLOAD_DIR + '/worker-' + str(TASK_INDEX) + '.txt', 'w').close()  # 清空文件

    with strategy.scope():
        # load data
        ds_train = Input_fn().batch(BATCH_SIZE).repeat()
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.DATA
        ds_train = ds_train.with_options(options)

        model_vgg = keras.applications.vgg16.VGG16(include_top=False, input_shape=(SIZE, SIZE, 3), weights=None)
        classes = Classes
        for layer in model_vgg.layers:
            layer.trainable = False
        model = Flatten()(model_vgg.output)
        model = Dense(4096, activation='relu', name='fc1')(model)
        model = Dropout(0.5)(model)
        model = Dense(4096, activation='relu', name='fc2')(model)
        model = Dropout(0.5)(model)
        model = Dense(classes_num, activation='softmax', name='prediction')(model)
        finetuned_model = Model(model_vgg.input, model)
        finetuned_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy',
                                metrics=['accuracy'])
        finetuned_model.classes = classes

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
    finetuned_model.fit(ds_train, steps_per_epoch=num_train_steps, epochs=EPOCHS,
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

    finetuned_model.save(model_path)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=False, default=16)
    parser.add_argument('--learning_rate', type=float, required=False, default=0.0001)
    parser.add_argument('--epochs', type=int, required=False, default=100)
    parser.add_argument('--workload_name', type=str, required=False, default='vgg')
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