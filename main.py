# -*- coding: utf-8 -*-
import argparse
import glob
import os
import chainer
import numpy as np
import functools
from chainer.backends import cuda

from dataset import SingleImageDataset
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser()
parser.add_argument("--datapath", type=str, default='./pictures',
                                            help='Please prepare HR images in this directory')
parser.add_argument("--insize", type=int,default=32)
parser.add_argument("--scale", type=int,default=4)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--batchsize", type=int, default=16)
parser.add_argument("--pretrained_generator")
args = parser.parse_args()

if args.gpu >= 0:
    #chainer.cuda.check_cuda_available()
    #chainer.cuda.get_device(args.gpu).use()
    xp = chainer.cuda.cupy
else:
    xp = np


class default_conv2d(chainer.Link):
    def __init__(self, in_channels=64, out_channels=64, ksize=3, stride=1, pad=1, activate=chainer.functions.leaky_relu):
        super(default_conv2d, self).__init__()
        with self.init_scope():
            self.conv=chainer.links.Convolution2D(in_channels, out_channels, ksize=ksize, stride=stride, pad=pad)
            self.activate=activate

    def __call__(self, x):
        x1=self.conv(x)
        return self.activate(x1)

class MODEL(chainer.Chain):
    def __init__(self):
        super(MODEL, self).__init__()
        with self.init_scope():
            self.first=default_conv2d(3, 64, ksize=5, stride=1, pad=2)
            self.conv1=default_conv2d(64, 64, ksize=3, stride=1, pad=1)
            self.deconv1=chainer.links.Deconvolution2D(64,3,ksize=4,stride=4)

    def __call__(self, x: chainer.Variable, test=False):
        
        x0 = self.first(x)

        x0=self.conv1(x0)

        h=self.deconv1(x0)

        return h

def train():
    dataset = SingleImageDataset(paths=args.datapath, in_size=args.insize, scale=args.scale)

    iterator = chainer.iterators.MultithreadIterator(dataset, batch_size=args.batchsize, repeat=True, shuffle=True)
    # iterator = chainer.iterators.SerialIterator(dataset, batch_size=args.batchsize, repeat=True, shuffle=True)

    generator = MODEL()
    if args.pretrained_generator is not None:
        chainer.serializers.load_npz(args.pretrained_generator, generator)
    if args.gpu >= 0:
        generator.to_gpu(args.gpu)

    optimizer_generator = chainer.optimizers.Adam(alpha=1e-4)
    optimizer_generator.setup(generator)
    optimizer_generator.use_cleargrads()

    # updater = chainer.training.updaters.StandardUpdater(iterator, optimizer_generator, device=None)#args.gpu)

    # trainer = chainer.training.Trainer(updater, (200, 'epoch'), out='result')
    # trainer.run()

    step = 0
    sum_loss_generator = 0
    for zipped_batch in iterator:
        lr = chainer.Variable(xp.array([zipped[0] for zipped in zipped_batch]))
        hr = chainer.Variable(xp.array([zipped[1] for zipped in zipped_batch]))

        sr = generator(lr)

        
        loss_generator = chainer.functions.mean_absolute_error(
                sr,
                hr
        )
        #optimizer_generator.zero_grads()
        loss_generator.backward()
        optimizer_generator.update()
        loss_g=chainer.cuda.to_cpu(loss_generator.data)
        sum_loss_generator += loss_g

        report_span = 1
        save_span = 500
        step += 1
        if step % report_span == 0:
            sum_loss_generator = 0
            print("Step: {}".format(step))
            print("loss_generator: {}".format(loss_g))
        if step % save_span == 0:
            chainer.serializers.save_npz(
                os.path.join('checkpoint', "generator_model_{}.npz".format(step)), generator)


if __name__=='__main__':
    train()