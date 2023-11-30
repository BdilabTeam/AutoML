#!/user/bin/env python
# -*- coding: UTF-8 -*-

"""
@author: liu wenjing
@create: 2021/10/26 19:36
"""
import math, json, os, sys
import pathlib
import argparse
# import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import Embedding
from model import *
from datetime import datetime
import tensorflow_text as tf_text
import pynvml
import yaml

tf.config.experimental_run_functions_eagerly(True)
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

# embedding_dim = 128
units = 256  # 隐藏层单元数
max_vocab_size = 5000
BATCH_SIZE = 16
EPOCHS = 3
embedding_dim = 128
DATA_DIR = '../../dataset/'
path_to_file = pathlib.Path(DATA_DIR)/'spa-eng/spa.txt'

# embedding = os.path.join(DATA_DIR, '2')
# hub_layer = hub.load(embedding)

def load_data(path):
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines()
    pairs = [line.split('\t') for line in lines]
    inp = [inp for targ, inp in pairs]
    targ = [targ for targ, inp in pairs]
    return targ, inp


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

BUFFER_SIZE = 1000
targ, inp = load_data(path_to_file)
input_text_processor = preprocessing.TextVectorization(standardize=tf_lower_and_split_punct, max_tokens=max_vocab_size)
input_text_processor.adapt(targ)
output_text_processor = preprocessing.TextVectorization(standardize=tf_lower_and_split_punct, max_tokens=max_vocab_size)
output_text_processor.adapt(inp)

# input_tokens = input_text_processor(targ)
# target_tokens = input_text_processor(inp)
#
# # embedding = Embedding(max_vocab_size, embedding_dim)
# targ_examples = input_tokens
# inp_examples = target_tokens


# length = len(targ)
# targ_examples = hub_layer(targ)
# targ_examples = tf.expand_dims(targ_examples, 1)
# inp_examples = hub_layer(inp)
# inp_examples = tf.expand_dims(inp_examples, 1)
# print(type(targ_examples), targ_examples.shape, targ_examples[0])
# print(type(inp_examples), inp_examples.shape, inp_examples[0])
dataset = tf.data.Dataset.from_tensor_slices((inp, targ))
dataset = dataset.batch(BATCH_SIZE)


translator = TrainTranslator(embedding_dim, units, input_text_processor=input_text_processor, output_text_processor=output_text_processor)
translator.compile(optimizer=tf.optimizers.Adam(), loss=MaskedLoss())
translator.fit(dataset, epochs=EPOCHS)




