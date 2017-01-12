import argparse
import numpy as np
import scipy.misc
from chainer import cuda
from chainer import serializers, datasets
from models import Inception
import inception_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--n-samples', type=int, default=200)
    # parser.add_argument('--params', type=str, default='inception_score.model')
    parser.add_argument('--params', type=str, default='test.model')
    return parser.parse_args()


def load_model(args):
    gpu = args.gpu
    params = args.params

    model = Inception()
    serializers.load_hdf5(params, model)

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    return model


def load_ims(args, scale=255.0):
    n_samples = args.n_samples

    ims, _ = datasets.get_cifar10(ndim=3, withlabel=False, scale=scale)
    ims = ims[:n_samples]

    print('Preprocessing images...')

    # Resize images to 299x299 which is the expected dimension of the
    # inception module
    ims_resized = np.empty((ims.shape[0], ims.shape[1], 299, 299))
    for i, im in enumerate(ims):
        im = im.transpose((1, 2, 0))
        im = scipy.misc.imresize(im, (299, 299), interp='bilinear')
        ims_resized[i] = im.transpose((2, 0, 1))

    return ims_resized


def main(args):
    model = load_model(args)
    ims = load_ims(args)

    mean, std = inception_score.inception_score(model, ims)

    print('Inception score mean:', mean)
    print('Inception score std:', std)


if __name__ == '__main__':
    args = parse_args()
    main(args)
