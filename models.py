import chainer
from chainer import functions as F
from chainer import links as L


class Mixed(chainer.Chain):
    def __init__(self, trunk):
        super().__init__()
        for name, link in trunk:
            self.add_link(name, link)
        self.trunk = trunk

    def __call__(self, x, test=True):
        hs = []
        for name, _ in self.trunk:
            h = getattr(self, name)(x, test=test)
            hs.append(h)
        return F.concat(hs)


class Tower(chainer.Chain):
    def __init__(self, trunk):
        super().__init__()
        for name, link in trunk:
            if not name.startswith('_'):
                self.add_link(name, link)
        self.trunk = trunk

    def __call__(self, x, test=True):
        h = x
        for name, f in self.trunk:
            if not name.startswith('_'):  # Link
                if 'bn' in name:
                    h = getattr(self, name)(h, test=test)
                else:
                    h = getattr(self, name)(h)
            else:  # AveragePooling2D, MaxPooling2D or ReLU
                h = f(h)
        return h


class Inception(chainer.Chain):
    def __init__(self):
        super().__init__(
            conv=L.Convolution2D(3, 32, 3, stride=2, pad=0),
            conv_1=L.Convolution2D(32, 32, 3, stride=1, pad=0),
            conv_2=L.Convolution2D(32, 64, 3, stride=1, pad=1),
            conv_3=L.Convolution2D(64, 80, 1, stride=1, pad=0),
            conv_4=L.Convolution2D(80, 192, 3, stride=1, pad=0),
            bn_conv=L.BatchNormalization(32),
            bn_conv_1=L.BatchNormalization(32),
            bn_conv_2=L.BatchNormalization(64),
            bn_conv_3=L.BatchNormalization(80),
            bn_conv_4=L.BatchNormalization(192),
            mixed=Mixed([
                ('conv', Tower([
                    ('conv', L.Convolution2D(192, 64, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(64)),
                    ('_relu', F.ReLU())
                ])),
                ('tower', Tower([
                    ('conv', L.Convolution2D(192, 48, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(48)),
                    ('_relu', F.ReLU()),
                    ('conv_1', L.Convolution2D(48, 64, 5, stride=1, pad=2)),
                    ('bn_conv_1', L.BatchNormalization(64)),
                    ('_relu_1', F.ReLU())
                ])),
                ('tower_1', Tower([
                    ('conv', L.Convolution2D(192, 64, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(64)),
                    ('_relu', F.ReLU()),
                    ('conv_1', L.Convolution2D(64, 96, 3, stride=1, pad=1)),
                    ('bn_conv_1', L.BatchNormalization(96)),
                    ('_relu_1', F.ReLU()),
                    ('conv_2', L.Convolution2D(96, 96, 3, stride=1, pad=1)),
                    ('bn_conv_2', L.BatchNormalization(96)),
                    ('_relu_2', F.ReLU())
                ])),
                ('tower_2', Tower([
                    ('_pooling', F.AveragePooling2D(3, 1, pad=1)),
                    ('conv', L.Convolution2D(192, 32, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(32)),
                    ('_relu', F.ReLU())
                ]))
            ]),
            mixed_1=Mixed([
                ('conv', Tower([
                    ('conv', L.Convolution2D(256, 64, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(64)),
                    ('_relu', F.ReLU())
                ])),
                ('tower', Tower([
                    ('conv', L.Convolution2D(256, 48, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(48)),
                    ('_relu', F.ReLU()),
                    ('conv_1', L.Convolution2D(48, 64, 5, stride=1, pad=2)),
                    ('bn_conv_1', L.BatchNormalization(64)),
                    ('_relu_1', F.ReLU())
                ])),
                ('tower_1', Tower([
                    ('conv', L.Convolution2D(256, 64, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(64)),
                    ('_relu', F.ReLU()),
                    ('conv_1', L.Convolution2D(64, 96, 3, stride=1, pad=1)),
                    ('bn_conv_1', L.BatchNormalization(96)),
                    ('_relu_1', F.ReLU()),
                    ('conv_2', L.Convolution2D(96, 96, 3, stride=1, pad=1)),
                    ('bn_conv_2', L.BatchNormalization(96)),
                    ('_relu_2', F.ReLU())
                ])),
                ('tower_2', Tower([
                    ('_pooling', F.AveragePooling2D(3, 1, pad=1)),
                    ('conv', L.Convolution2D(256, 64, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(64)),
                    ('_relu', F.ReLU())
                ]))
            ]),
            mixed_2=Mixed([
                ('conv', Tower([
                    ('conv', L.Convolution2D(288, 64, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(64)),
                    ('_relu', F.ReLU())
                ])),
                ('tower', Tower([
                    ('conv', L.Convolution2D(288, 48, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(48)),
                    ('_relu', F.ReLU()),
                    ('conv_1', L.Convolution2D(48, 64, 5, stride=1, pad=2)),
                    ('bn_conv_1', L.BatchNormalization(64)),
                    ('_relu_1', F.ReLU())
                ])),
                ('tower_1', Tower([
                    ('conv', L.Convolution2D(288, 64, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(64)),
                    ('_relu', F.ReLU()),
                    ('conv_1', L.Convolution2D(64, 96, 3, stride=1, pad=1)),
                    ('bn_conv_1', L.BatchNormalization(96)),
                    ('_relu_1', F.ReLU()),
                    ('conv_2', L.Convolution2D(96, 96, 3, stride=1, pad=1)),
                    ('bn_conv_2', L.BatchNormalization(96)),
                    ('_relu_2', F.ReLU())
                ])),
                ('tower_2', Tower([
                    ('_pooling', F.AveragePooling2D(3, 1, pad=1)),
                    ('conv', L.Convolution2D(288, 64, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(64)),
                    ('_relu', F.ReLU())
                ]))
            ]),
            mixed_3=Mixed([
                ('conv', Tower([
                    ('conv', L.Convolution2D(288, 384, 3, stride=2, pad=0)),
                    ('bn_conv', L.BatchNormalization(384)),
                    ('_relu', F.ReLU())
                ])),
                ('tower', Tower([
                    ('conv', L.Convolution2D(288, 64, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(64)),
                    ('_relu', F.ReLU()),
                    ('conv_1', L.Convolution2D(64, 96, 3, stride=1, pad=1)),
                    ('bn_conv_1', L.BatchNormalization(96)),
                    ('_relu_1', F.ReLU()),
                    ('conv_2', L.Convolution2D(96, 96, 3, stride=2, pad=0)),
                    ('bn_conv_2', L.BatchNormalization(96)),
                    ('_relu_2', F.ReLU())
                ])),
                ('pool', Tower([
                    ('_pooling', F.MaxPooling2D(3, 2, pad=0))
                ]))
            ]),
            mixed_4=Mixed([
                ('conv', Tower([
                    ('conv', L.Convolution2D(768, 192, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(192)),
                    ('_relu', F.ReLU())
                ])),
                ('tower', Tower([
                    ('conv', L.Convolution2D(768, 128, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(128)),
                    ('_relu', F.ReLU()),
                    ('conv_1', L.Convolution2D(128, 128, (1, 7), stride=1, pad=(0, 3))),
                    ('bn_conv_1', L.BatchNormalization(128)),
                    ('_relu_1', F.ReLU()),
                    ('conv_2', L.Convolution2D(128, 192, (7, 1), stride=1, pad=(3, 0))),
                    ('bn_conv_2', L.BatchNormalization(192)),
                    ('_relu_2', F.ReLU())
                ])),
                ('tower_1', Tower([
                    ('conv', L.Convolution2D(768, 128, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(128)),
                    ('_relu', F.ReLU()),
                    ('conv_1', L.Convolution2D(128, 128, (7, 1), stride=1, pad=(3, 0))),
                    ('bn_conv_1', L.BatchNormalization(128)),
                    ('_relu_1', F.ReLU()),
                    ('conv_2', L.Convolution2D(128, 128, (1, 7), stride=1, pad=(0, 3))),
                    ('bn_conv_2', L.BatchNormalization(128)),
                    ('_relu_2', F.ReLU()),
                    ('conv_3', L.Convolution2D(128, 128, (7, 1), stride=1, pad=(3, 0))),
                    ('bn_conv_3', L.BatchNormalization(128)),
                    ('_relu_3', F.ReLU()),
                    ('conv_4', L.Convolution2D(128, 192, (1, 7), stride=1, pad=(0, 3))),
                    ('bn_conv_4', L.BatchNormalization(192)),
                    ('_relu_4', F.ReLU())
                ])),
                ('tower_2', Tower([
                    ('_pooling', F.AveragePooling2D(3, 1, pad=1)),
                    ('conv', L.Convolution2D(768, 192, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(192)),
                    ('_relu', F.ReLU())
                ]))
            ]),
            mixed_5=Mixed([
                ('conv', Tower([
                    ('conv', L.Convolution2D(768, 192, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(192)),
                    ('_relu', F.ReLU())
                ])),
                ('tower', Tower([
                    ('conv', L.Convolution2D(768, 160, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(160)),
                    ('_relu', F.ReLU()),
                    ('conv_1', L.Convolution2D(160, 160, (1, 7), stride=1, pad=(0, 3))),
                    ('bn_conv_1', L.BatchNormalization(160)),
                    ('_relu_1', F.ReLU()),
                    ('conv_2', L.Convolution2D(160, 192, (7, 1), stride=1, pad=(3, 0))),
                    ('bn_conv_2', L.BatchNormalization(192)),
                    ('_relu_2', F.ReLU())
                ])),
                ('tower_1', Tower([
                    ('conv', L.Convolution2D(768, 160, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(160)),
                    ('_relu', F.ReLU()),
                    ('conv_1', L.Convolution2D(160, 160, (7, 1), stride=1, pad=(3, 0))),
                    ('bn_conv_1', L.BatchNormalization(160)),
                    ('_relu_1', F.ReLU()),
                    ('conv_2', L.Convolution2D(160, 160, (1, 7), stride=1, pad=(0, 3))),
                    ('bn_conv_2', L.BatchNormalization(160)),
                    ('_relu_2', F.ReLU()),
                    ('conv_3', L.Convolution2D(160, 160, (7, 1), stride=1, pad=(3, 0))),
                    ('bn_conv_3', L.BatchNormalization(160)),
                    ('_relu_3', F.ReLU()),
                    ('conv_4', L.Convolution2D(160, 192, (1, 7), stride=1, pad=(0, 3))),
                    ('bn_conv_4', L.BatchNormalization(192)),
                    ('_relu_4', F.ReLU())
                ])),
                ('tower_2', Tower([
                    ('_pooling', F.AveragePooling2D(3, 1, pad=1)),
                    ('conv', L.Convolution2D(768, 192, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(192)),
                    ('_relu', F.ReLU())
                ]))
            ]),
            mixed_6=Mixed([
                ('conv', Tower([
                    ('conv', L.Convolution2D(768, 192, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(192)),
                    ('_relu', F.ReLU())
                ])),
                ('tower', Tower([
                    ('conv', L.Convolution2D(768, 160, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(160)),
                    ('_relu', F.ReLU()),
                    ('conv_1', L.Convolution2D(160, 160, (1, 7), stride=1, pad=(0, 3))),
                    ('bn_conv_1', L.BatchNormalization(160)),
                    ('_relu_1', F.ReLU()),
                    ('conv_2', L.Convolution2D(160, 192, (7, 1), stride=1, pad=(3, 0))),
                    ('bn_conv_2', L.BatchNormalization(192)),
                    ('_relu_2', F.ReLU())
                ])),
                ('tower_1', Tower([
                    ('conv', L.Convolution2D(768, 160, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(160)),
                    ('_relu', F.ReLU()),
                    ('conv_1', L.Convolution2D(160, 160, (7, 1), stride=1, pad=(3, 0))),
                    ('bn_conv_1', L.BatchNormalization(160)),
                    ('_relu_1', F.ReLU()),
                    ('conv_2', L.Convolution2D(160, 160, (1, 7), stride=1, pad=(0, 3))),
                    ('bn_conv_2', L.BatchNormalization(160)),
                    ('_relu_2', F.ReLU()),
                    ('conv_3', L.Convolution2D(160, 160, (7, 1), stride=1, pad=(3, 0))),
                    ('bn_conv_3', L.BatchNormalization(160)),
                    ('_relu_3', F.ReLU()),
                    ('conv_4', L.Convolution2D(160, 192, (1, 7), stride=1, pad=(0, 3))),
                    ('bn_conv_4', L.BatchNormalization(192)),
                    ('_relu_4', F.ReLU())
                ])),
                ('tower_2', Tower([
                    ('_pooling', F.AveragePooling2D(3, 1, pad=1)),
                    ('conv', L.Convolution2D(768, 192, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(192)),
                    ('_relu', F.ReLU())
                ]))
            ]),
            mixed_7=Mixed([
                ('conv', Tower([
                    ('conv', L.Convolution2D(768, 192, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(192)),
                    ('_relu', F.ReLU())
                ])),
                ('tower', Tower([
                    ('conv', L.Convolution2D(768, 192, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(192)),
                    ('_relu', F.ReLU()),
                    ('conv_1', L.Convolution2D(192, 192, (1, 7), stride=1, pad=(0, 3))),
                    ('bn_conv_1', L.BatchNormalization(192)),
                    ('_relu_1', F.ReLU()),
                    ('conv_2', L.Convolution2D(192, 192, (7, 1), stride=1, pad=(3, 0))),
                    ('bn_conv_2', L.BatchNormalization(192)),
                    ('_relu_2', F.ReLU())
                ])),
                ('tower_1', Tower([
                    ('conv', L.Convolution2D(768, 192, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(192)),
                    ('_relu', F.ReLU()),
                    ('conv_1', L.Convolution2D(192, 192, (7, 1), stride=1, pad=(3, 0))),
                    ('bn_conv_1', L.BatchNormalization(192)),
                    ('_relu_1', F.ReLU()),
                    ('conv_2', L.Convolution2D(192, 192, (1, 7), stride=1, pad=(0, 3))),
                    ('bn_conv_2', L.BatchNormalization(192)),
                    ('_relu_2', F.ReLU()),
                    ('conv_3', L.Convolution2D(192, 192, (7, 1), stride=1, pad=(3, 0))),
                    ('bn_conv_3', L.BatchNormalization(192)),
                    ('_relu_3', F.ReLU()),
                    ('conv_4', L.Convolution2D(192, 192, (1, 7), stride=1, pad=(0, 3))),
                    ('bn_conv_4', L.BatchNormalization(192)),
                    ('_relu_4', F.ReLU())
                ])),
                ('tower_2', Tower([
                    ('_pooling', F.AveragePooling2D(3, 1, pad=1)),
                    ('conv', L.Convolution2D(768, 192, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(192)),
                    ('_relu', F.ReLU())
                ]))
            ]),
            mixed_8=Mixed([
                ('tower', Tower([
                    ('conv', L.Convolution2D(768, 192, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(192)),
                    ('_relu', F.ReLU()),
                    ('conv_1', L.Convolution2D(192, 320, 3, stride=2, pad=0)),
                    ('bn_conv_1', L.BatchNormalization(320)),
                    ('_relu_1', F.ReLU())
                ])),
                ('tower_1', Tower([
                    ('conv', L.Convolution2D(768, 192, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(192)),
                    ('_relu', F.ReLU()),
                    ('conv_1', L.Convolution2D(192, 192, (1, 7), stride=1, pad=(0, 3))),
                    ('bn_conv_1', L.BatchNormalization(192)),
                    ('_relu_1', F.ReLU()),
                    ('conv_2', L.Convolution2D(192, 192, (7, 1), stride=1, pad=(3, 0))),
                    ('bn_conv_2', L.BatchNormalization(192)),
                    ('_relu_2', F.ReLU()),
                    ('conv_3', L.Convolution2D(192, 192, 3, stride=2, pad=0)),
                    ('bn_conv_3', L.BatchNormalization(192)),
                    ('_relu_3', F.ReLU())
                ])),
                ('pool', Tower([
                    ('_pooling', F.MaxPooling2D(3, 2, pad=0))
                ]))
            ]),
            mixed_9=Mixed([
                ('conv', Tower([
                    ('conv', L.Convolution2D(1280, 320, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(320)),
                    ('_relu', F.ReLU()),
                ])),
                ('tower', Tower([
                    ('conv', L.Convolution2D(1280, 384, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(384)),
                    ('_relu', F.ReLU()),
                    ('mixed', Mixed([
                        ('conv', Tower([
                            ('conv', L.Convolution2D(384, 384, (1, 3), stride=1, pad=(0, 1))),
                            ('bn_conv', L.BatchNormalization(384)),
                            ('_relu', F.ReLU()),
                        ])),
                        ('conv_1', Tower([
                            ('conv_1', L.Convolution2D(384, 384, (3, 1), stride=1, pad=(1, 0))),
                            ('bn_conv_1', L.BatchNormalization(384)),
                            ('_relu_1', F.ReLU()),
                        ]))
                    ]))
                ])),
                ('tower_1', Tower([
                    ('conv', L.Convolution2D(1280, 448, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(448)),
                    ('_relu', F.ReLU()),
                    ('conv_1', L.Convolution2D(448, 384, 3, stride=1, pad=1)),
                    ('bn_conv_1', L.BatchNormalization(384)),
                    ('_relu_1', F.ReLU()),
                    ('mixed', Mixed([
                        ('conv', Tower([
                            ('conv', L.Convolution2D(384, 384, (1, 3), stride=1, pad=(0, 1))),
                            ('bn_conv', L.BatchNormalization(384)),
                            ('_relu', F.ReLU()),
                        ])),
                        ('conv_1', Tower([
                            ('conv_1', L.Convolution2D(384, 384, (3, 1), stride=1, pad=(1, 0))),
                            ('bn_conv_1', L.BatchNormalization(384)),
                            ('_relu_1', F.ReLU()),
                        ]))
                    ]))
                ])),
                ('tower_2', Tower([
                    ('_pooling', F.AveragePooling2D(3, 1, pad=1)),
                    ('conv', L.Convolution2D(1280, 192, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(192)),
                    ('_relu', F.ReLU())
                ]))
            ]),
            mixed_10=Mixed([
                ('conv', Tower([
                    ('conv', L.Convolution2D(2048, 320, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(320)),
                    ('_relu', F.ReLU()),
                ])),
                ('tower', Tower([
                    ('conv', L.Convolution2D(2048, 384, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(384)),
                    ('_relu', F.ReLU()),
                    ('mixed', Mixed([
                        ('conv', Tower([
                            ('conv', L.Convolution2D(384, 384, (1, 3), stride=1, pad=(0, 1))),
                            ('bn_conv', L.BatchNormalization(384)),
                            ('_relu', F.ReLU()),
                        ])),
                        ('conv_1', Tower([
                            ('conv_1', L.Convolution2D(384, 384, (3, 1), stride=1, pad=(1, 0))),
                            ('bn_conv_1', L.BatchNormalization(384)),
                            ('_relu_1', F.ReLU()),
                        ]))
                    ]))
                ])),
                ('tower_1', Tower([
                    ('conv', L.Convolution2D(2048, 448, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(448)),
                    ('_relu', F.ReLU()),
                    ('conv_1', L.Convolution2D(448, 384, 3, stride=1, pad=1)),
                    ('bn_conv_1', L.BatchNormalization(384)),
                    ('_relu_1', F.ReLU()),
                    ('mixed', Mixed([
                        ('conv', Tower([
                            ('conv', L.Convolution2D(384, 384, (1, 3), stride=1, pad=(0, 1))),
                            ('bn_conv', L.BatchNormalization(384)),
                            ('_relu', F.ReLU())
                        ])),
                        ('conv_1', Tower([
                            ('conv_1', L.Convolution2D(384, 384, (3, 1), stride=1, pad=(1, 0))),
                            ('bn_conv_1', L.BatchNormalization(384)),
                            ('_relu_1', F.ReLU())
                        ]))
                    ]))
                ])),
                ('tower_2', Tower([
                    ('_pooling', F.MaxPooling2D(3, 1, pad=1)),
                    ('conv', L.Convolution2D(2048, 192, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(192)),
                    ('_relu', F.ReLU())
                ]))
            ]),
            logit=L.Linear(2048, 1008)
        )

    def __call__(self, x, test=True):
        """Input dims are (batch_size, 3, 299, 299)."""

        assert x.shape[1:] == (3, 299, 299)

        x -= 128.0
        x *= 0.0078125

        h = F.relu(self.bn_conv(self.conv(x), test=test))
        assert h.shape[1:] == (32, 149, 149)

        h = F.relu(self.bn_conv_1(self.conv_1(h), test=test))
        assert h.shape[1:] == (32, 147, 147)

        h = F.relu(self.bn_conv_2(self.conv_2(h), test=test))
        assert h.shape[1:] == (64, 147, 147)

        h = F.max_pooling_2d(h, 3, stride=2, pad=0)
        assert h.shape[1:] == (64, 73, 73)

        h = F.relu(self.bn_conv_3(self.conv_3(h), test=test))
        assert h.shape[1:] == (80, 73, 73)

        h = F.relu(self.bn_conv_4(self.conv_4(h), test=test))
        assert h.shape[1:] == (192, 71, 71)

        h = F.max_pooling_2d(h, 3, stride=2, pad=0)
        assert h.shape[1:] == (192, 35, 35)

        h = self.mixed(h, test=test)
        assert h.shape[1:] == (256, 35, 35)

        h = self.mixed_1(h, test=test)
        assert h.shape[1:] == (288, 35, 35)

        h = self.mixed_2(h, test=test)
        assert h.shape[1:] == (288, 35, 35)

        h = self.mixed_3(h, test=test)
        assert h.shape[1:] == (768, 17, 17)

        h = self.mixed_4(h, test=test)
        assert h.shape[1:] == (768, 17, 17)

        h = self.mixed_5(h, test=test)
        assert h.shape[1:] == (768, 17, 17)

        h = self.mixed_6(h, test=test)
        assert h.shape[1:] == (768, 17, 17)

        h = self.mixed_7(h, test=test)
        assert h.shape[1:] == (768, 17, 17)

        h = self.mixed_8(h, test=test)
        assert h.shape[1:] == (1280, 8, 8)

        h = self.mixed_9(h, test=test)
        assert h.shape[1:] == (2048, 8, 8)

        h = self.mixed_10(h, test=test)
        assert h.shape[1:] == (2048, 8, 8)

        h = F.average_pooling_2d(h, 8, 1)
        assert h.shape[1:] == (2048, 1, 1)

        h = F.reshape(h, (-1, 2048))
        h = self.logit(h)
        h = F.softmax(h)

        assert h.shape[1:] == (1008,)

        return h
