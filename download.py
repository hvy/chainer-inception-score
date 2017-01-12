# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import tarfile

import argparse
import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
import scipy.misc
import math
import sys

from chainer import serializers
from models import Inception

MODEL_DIR = '/tmp/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile', type=str, default='test.model')
    return parser.parse_args()


# Works with an arbitrary minibatch size.
def copy_conv(sess, tftensor, layer):
    W = sess.graph.get_tensor_by_name('{}/conv2d_params:0'.format(tftensor)).eval()
    W = W.transpose((3, 2, 0, 1))

    assert W.shape == layer.W.data.shape

    layer.W.data = W


def copy_bn(sess, tftensor, layer):
    gamma = sess.graph.get_tensor_by_name('{}/gamma:0'.format(tftensor)).eval()
    beta = sess.graph.get_tensor_by_name('{}/beta:0'.format(tftensor)).eval()
    avg_mean = sess.graph.get_tensor_by_name('{}/moving_mean:0'.format(tftensor)).eval()
    avg_var = sess.graph.get_tensor_by_name('{}/moving_variance:0'.format(tftensor)).eval()
    eps = sess.graph.get_operation_by_name(tftensor).get_attr('variance_epsilon')

    assert layer.beta.data.shape == beta.shape
    assert layer.gamma.data.shape == gamma.shape
    assert layer.avg_mean.shape == avg_mean.shape
    assert layer.avg_var.shape == avg_var.shape
    assert eps > 0.0

    layer.beta.data = beta
    layer.gamma.data = gamma
    layer.avg_mean = avg_mean
    layer.avg_var = avg_var
    layer.eps = eps


def copy_inception(sess, net):
    # Copy parameters from the TensorFlow model the a Chainer model
    print('Copying first layers ...')
    copy_conv(sess, 'conv', net.conv)
    copy_bn(sess, 'conv/batchnorm', net.bn_conv)
    copy_conv(sess, 'conv_1', net.conv_1)
    copy_bn(sess, 'conv_1/batchnorm', net.bn_conv_1)
    copy_conv(sess, 'conv_2', net.conv_2)
    copy_bn(sess, 'conv_2/batchnorm', net.bn_conv_2)
    copy_conv(sess, 'conv_3', net.conv_3)
    copy_bn(sess, 'conv_3/batchnorm', net.bn_conv_3)
    copy_conv(sess, 'conv_4', net.conv_4)
    copy_bn(sess, 'conv_4/batchnorm', net.bn_conv_4)

    for m in ['mixed', 'mixed_1', 'mixed_2']:
        print('Copying ', m, '...')
        copy_conv(sess, '{}/conv'.format(m), getattr(net, m).conv.conv)
        copy_bn(sess, '{}/conv/batchnorm'.format(m), getattr(net, m).conv.bn_conv)

        for t in ['tower', 'tower_1', 'tower_2']:
            copy_conv(sess, '{}/{}/conv'.format(m, t), getattr(getattr(net, m), t).conv)
            copy_bn(sess, '{}/{}/conv/batchnorm'.format(m, t), getattr(getattr(net, m), t).bn_conv)

            if t == 'tower' or t == 'tower_1':
                copy_conv(sess, '{}/{}/conv_1'.format(m, t), getattr(getattr(net, m), t).conv_1)
                copy_bn(sess, '{}/{}/conv_1/batchnorm'.format(m, t), getattr(getattr(net, m), t).bn_conv_1)

            if t == 'tower_1':
                copy_conv(sess, '{}/{}/conv_2'.format(m, t), getattr(getattr(net, m), t).conv_2)
                copy_bn(sess, '{}/{}/conv_2/batchnorm'.format(m, t), getattr(getattr(net, m), t).bn_conv_2)

    for m in ['mixed_3']:
        print('Copying ', m, '...')
        copy_conv(sess, '{}/conv'.format(m), getattr(net, m).conv.conv)
        copy_bn(sess, '{}/conv/batchnorm'.format(m), getattr(net, m).conv.bn_conv)

        for t in ['tower']:
            copy_conv(sess, '{}/{}/conv'.format(m, t), getattr(getattr(net, m), t).conv)
            copy_bn(sess, '{}/{}/conv/batchnorm'.format(m, t), getattr(getattr(net, m), t).bn_conv)
            copy_conv(sess, '{}/{}/conv_1'.format(m, t), getattr(getattr(net, m), t).conv_1)
            copy_bn(sess, '{}/{}/conv_1/batchnorm'.format(m, t), getattr(getattr(net, m), t).bn_conv_1)
            copy_conv(sess, '{}/{}/conv_2'.format(m, t), getattr(getattr(net, m), t).conv_2)
            copy_bn(sess, '{}/{}/conv_2/batchnorm'.format(m, t), getattr(getattr(net, m), t).bn_conv_2)

    for m in ['mixed_4', 'mixed_5', 'mixed_6', 'mixed_7']:
        print('Copying ', m, '...')
        copy_conv(sess, '{}/conv'.format(m), getattr(net, m).conv.conv)
        copy_bn(sess, '{}/conv/batchnorm'.format(m), getattr(net, m).conv.bn_conv)

        for t in ['tower', 'tower_1', 'tower_2']:
            copy_conv(sess, '{}/{}/conv'.format(m, t), getattr(getattr(net, m), t).conv)
            copy_bn(sess, '{}/{}/conv/batchnorm'.format(m, t), getattr(getattr(net, m), t).bn_conv)

            if t == 'tower' or t == 'tower_1':
                copy_conv(sess, '{}/{}/conv_1'.format(m, t), getattr(getattr(net, m), t).conv_1)
                copy_bn(sess, '{}/{}/conv_1/batchnorm'.format(m, t), getattr(getattr(net, m), t).bn_conv_1)
                copy_conv(sess, '{}/{}/conv_2'.format(m, t), getattr(getattr(net, m), t).conv_2)
                copy_bn(sess, '{}/{}/conv_2/batchnorm'.format(m, t), getattr(getattr(net, m), t).bn_conv_2)

            if t == 'tower_1':
                copy_conv(sess, '{}/{}/conv_3'.format(m, t), getattr(getattr(net, m), t).conv_3)
                copy_bn(sess, '{}/{}/conv_3/batchnorm'.format(m, t), getattr(getattr(net, m), t).bn_conv_3)
                copy_conv(sess, '{}/{}/conv_4'.format(m, t), getattr(getattr(net, m), t).conv_4)
                copy_bn(sess, '{}/{}/conv_4/batchnorm'.format(m, t), getattr(getattr(net, m), t).bn_conv_4)

    for m in ['mixed_8']:
        print('Copying ', m, '...')
        for t in ['tower', 'tower_1']:
            copy_conv(sess, '{}/{}/conv'.format(m, t), getattr(getattr(net, m), t).conv)
            copy_bn(sess, '{}/{}/conv/batchnorm'.format(m, t), getattr(getattr(net, m), t).bn_conv)
            copy_conv(sess, '{}/{}/conv_1'.format(m, t), getattr(getattr(net, m), t).conv_1)
            copy_bn(sess, '{}/{}/conv_1/batchnorm'.format(m, t), getattr(getattr(net, m), t).bn_conv_1)

            if t == 'tower_1':
                copy_conv(sess, '{}/{}/conv_2'.format(m, t), getattr(getattr(net, m), t).conv_2)
                copy_bn(sess, '{}/{}/conv_2/batchnorm'.format(m, t), getattr(getattr(net, m), t).bn_conv_2)
                copy_conv(sess, '{}/{}/conv_3'.format(m, t), getattr(getattr(net, m), t).conv_3)
                copy_bn(sess, '{}/{}/conv_3/batchnorm'.format(m, t), getattr(getattr(net, m), t).bn_conv_3)

    for m in ['mixed_9', 'mixed_10']:
        print('Copying ', m, '...')
        copy_conv(sess, '{}/conv'.format(m), getattr(net, m).conv.conv)
        copy_bn(sess, '{}/conv/batchnorm'.format(m), getattr(net, m).conv.bn_conv)

        for t in ['tower', 'tower_1', 'tower_2']:
            copy_conv(sess, '{}/{}/conv'.format(m, t), getattr(getattr(net, m), t).conv)
            copy_bn(sess, '{}/{}/conv/batchnorm'.format(m, t), getattr(getattr(net, m), t).bn_conv)

            if t == 'tower' or t == 'tower_1':
                copy_conv(sess, '{}/{}/mixed/conv'.format(m, t), getattr(getattr(net, m), t).mixed.conv.conv)
                copy_bn(sess, '{}/{}/mixed/conv/batchnorm'.format(m, t), getattr(getattr(net, m), t).mixed.conv.bn_conv)
                copy_conv(sess, '{}/{}/mixed/conv_1'.format(m, t), getattr(getattr(net, m), t).mixed.conv_1.conv_1)
                copy_bn(sess, '{}/{}/mixed/conv_1/batchnorm'.format(m, t), getattr(getattr(net, m), t).mixed.conv_1.bn_conv_1)

            if t == 'tower_1':
                copy_conv(sess, '{}/{}/conv_1'.format(m, t), getattr(getattr(net, m), t).conv_1)
                copy_bn(sess, '{}/{}/conv_1/batchnorm'.format(m, t), getattr(getattr(net, m), t).bn_conv_1)

    print('Copying logit...')
    w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1].eval()
    b = sess.graph.get_tensor_by_name("softmax/biases:0").eval()

    assert w.T.shape == net.logit.W.shape
    assert b.shape == net.logit.b.shape

    net.logit.W.data = w.T
    net.logit.b.data = b


def copy_params(net):
  if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(MODEL_DIR, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')

  # Load downloaded model...
  tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
  with tf.gfile.FastGFile(os.path.join(
      MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

  # Write graph to file so that it can be visualized using TensorBoard
  # summary_writer = tf.summary.FileWriter('data', graph=graph_def)

  # Configure to dynamically allocate GPU memory
  config = tf.ConfigProto()
  config.gpu_options.allow_growth=True
  with tf.Session(config=config) as sess:
    copy_inception(sess, net)


def main(args):
    outfile = args.outfile
    model = Inception()

    copy_params(model)

    # TODO(hvy): Test score similarity with the original implementation

    # Save the copied model
    serializers.save_hdf5(outfile, model)


if __name__ == '__main__':
    args = parse_args()
    main(args)
