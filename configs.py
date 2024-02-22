import os
import socket

computer_name = socket.gethostname()


class VanillaVit:
    def __init__(self, epochs, batch_size, learning_rate):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate


class GaussianPixel:
    def __init__(self, epochs, batch_size, learning_rate):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        if computer_name == 'oasis':
            self.out_dir = f'results/celeba/gps'
            self.visualize = False
        else:
            self.out_dir = '/is/cluster/fast/pghosh/datasets/celebA/gps'
            self.visualize = False
            
        if os.environ.get('VISUALIZE', False):
            self.visualize = True


class CelebA:
    def __init__(self):
        self.classes = 40
        self.img_size = 128
        self.vanilla_vit = VanillaVit(epochs=50, batch_size=64, learning_rate=0.0001)
        self.gaussian_pixel = GaussianPixel(epochs=10_000, batch_size=80, learning_rate=0.0001)
        if computer_name == 'oasis':
            self.datapath = '/home/pghosh/repos/datasets/celeba'
            self.visualize = False
        else:
            self.datapath = '/is/cluster/fast/pghosh/datasets/celebA/'
            self.visualize = False

        if os.environ.get('VISUALIZE', False):
            self.visualize = True


celeba_config = CelebA()
