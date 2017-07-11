# Inception Score Module

Chainer implementation of the inception score module presented in [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498). The code is derived from the official OpenAI implementation at https://github.com/openai/improved-gan.

## Inception Score

A method proposed by Tim Salimans et al. (2016) in the above mentioned paper to evaluate generative models such as VAEs and GANs using a pre-trained classifier network and sampled images. 

It is based on the fact that good samples (images that look like images from the true data distribution) are expected to yield

1. low entropy p(y|x), i.e. high prediction confidence
2. high entropy p(y), i.e. highly varied predictions

where x is an image, p(y|x) is the inferred class label probabilities given x by the pre-trained Inception network and p(y) is the marginal distribution over all images.

The Inception score is defined as exp(E_x[KL(p(y|x) || p(y))])

## Usage

Download the pre-trained TensorFlow model and create a Chainer copy named *inception_score.model*.

```bash
python download.py --outfile inception_score.model
```

Load the pretrained Chainer model and compute the inception score for the CIFAR-10 dataset including both train and test images. To limit the number of images, use the `--samples` option.

```
python example.py --model inception_score.model
```

```
...
Batch size: 100
Total number of images: 60000
Total number of batches: 600
Running batch 1 / 600 ...
Running batch 2 / 600 ...
...
Running batch 600 / 600 ...
Inception score mean: 12.003619194030762
Inception score std: 0.10357429087162018
```

### Example Usage in Python

```python
import numpy as np
from chainer import serializers, datasets
from inception_score import Inception, inception_score

model = Inception()
serializers.load_hdf5('inception_score.model', model)

train, test = datasets.get_cifar10(ndim=3, withlabel=False, scale=255.0)
mean, std = inception_score(model, np.concatenate(train, test))

print('Inception score mean:', mean)
print('Inception score std:', std)
```

## Note

This implementation seems to yield slightly higher scores than the original implementation looking at the inception scores based on CIFAR-10, upsampled from (32, 32) to (299, 299) using bilinear interpolation.

|| Ours | Original |
| ------------- | ------------- | ------------- |
| Mean | 12.00 | 11.24 |
| Std  | 0.10 | 0.12 |
