import argparse
from chainer import cuda
from chainer import serializers, datasets
from inception_score import Inception, inception_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--samples', type=int, default=50000)
    parser.add_argument('--infile', type=str, default='inception_score.model')
    return parser.parse_args()


def load_model(args):
    gpu = args.gpu
    infile = args.infile

    model = Inception()
    serializers.load_hdf5(infile, model)

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    return model


def load_ims(args, scale=255.0):
    samples = args.samples

    ims, _ = datasets.get_cifar10(ndim=3, withlabel=False, scale=scale)
    ims = ims[:samples]

    return ims


def main(args):
    model = load_model(args)
    ims = load_ims(args)

    mean, std = inception_score(model, ims)

    print('Inception score mean:', mean)
    print('Inception score std:', std)


if __name__ == '__main__':
    args = parse_args()
    main(args)
