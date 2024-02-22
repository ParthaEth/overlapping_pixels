import os
import socket

computer_name = socket.gethostname()


class VanillaVit:
    def __init__(self, epochs, batch_size, learning_rate):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate


class NonUniformVit:
    def __init__(self, epochs, batch_size, learning_rate):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate


class GaussianPixel:
    def __init__(self, epochs, batch_size, learning_rate, pix_per_img):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        if computer_name == 'oasis':
            self.out_dir = f'results/celeba/gps'
            self.visualize = False
        else:
            self.out_dir = '/is/cluster/fast/pghosh/datasets/celebA/gps'
            self.visualize = False

        self.pix_per_img = pix_per_img

        if os.environ.get('VISUALIZE', False):
            self.visualize = True


class CelebA:
    def __init__(self, model_name):
        self.classes = 40
        self.img_size = 128
        self.model_name = model_name
        if model_name.lower() == 'vanilla_vit':
            self.vit_conf = VanillaVit(epochs=50, batch_size=64, learning_rate=0.0001)
        elif model_name.lower() == 'non_uniform_vit':
            self.vit_conf = NonUniformVit(epochs=50, batch_size=64, learning_rate=0.0001)
        else:
            raise ValueError(f'Unknown model name: {model_name}, possible values: vanilla_vit, non_uniform_vit')

        # TODO: Remove the following line and un comment the one after. Temoporary for testing
        self.gaussian_pixel = GaussianPixel(epochs=10_000, batch_size=80, learning_rate=0.0001, pix_per_img=64)
        # self.gaussian_pixel = GaussianPixel(epochs=10_000, batch_size=80, learning_rate=0.0001, pix_per_img=400)
        if computer_name == 'oasis':
            self.datapath = '/home/pghosh/repos/datasets/celeba'
            self.visualize = False
        else:
            self.datapath = '/is/cluster/fast/pghosh/datasets/celebA/'
            self.visualize = False

        if os.environ.get('VISUALIZE', False):
            self.visualize = True


# celeba_config = CelebA(model_name='vanilla_vit')
celeba_config = CelebA(model_name='non_uniform_vit')
