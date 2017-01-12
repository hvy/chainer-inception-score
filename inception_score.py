import math
import numpy as np
import cupy
from chainer import Variable, cuda


def inception_score(model, xs, batch_size=100, splits=10):
    n = xs.shape[0]
    n_batches = int(math.ceil(float(n) / float(batch_size)))

    xp = model.xp

    print('Number of images:', n)
    print('Batch size:', batch_size)
    print('Number of batches:', n_batches)

    ys = []
    for i in range(n_batches):
        x = xs[(i * batch_size):min((i + 1) * batch_size, n)]
        y = model(Variable(xp.array(x, dtype=xp.float32), volatile=True), test=True)
        y = y.data
        if isinstance(y, cupy.ndarray):
            y = cuda.to_cpu(y)
        ys.append(y)


    # Always compute the rest on the CPU to save GPU memory
    ys = np.concatenate(ys, 0)

    scores = []
    for i in range(splits):
      part = ys[(i * ys.shape[0] // splits):((i + 1) * ys.shape[0] // splits), :]
      kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
      kl = np.mean(np.sum(kl, 1))
      scores.append(np.exp(kl))

    return np.mean(scores), np.std(scores)
