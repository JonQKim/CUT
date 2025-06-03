import os
import argparse
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from modules.cut_model import CUT_model
from utils import create_dir, load_image


# source_shape = [512, 512, 3]
# source_shape = [256, 256, 3]
source_shape = [64, 64, 3]
target_shape = source_shape
cut_mode = 'fastcut'
# cut_mode='cut'
impl = 'ref'
gan_mode = 'lsgan'


tf.executing_eagerly()
cut = CUT_model(source_shape=source_shape, \
                target_shape=target_shape, \
                cut_mode=cut_mode, \
                gan_mode=gan_mode, \
                impl=impl)

netG = cut.netG
tf.saved_model.save(netG, 'netg')

converter = tf.lite.TFLiteConverter.from_saved_model('netg')
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()

with open('netg.tflite', 'wb') as f:
    f.write(tflite_model)




