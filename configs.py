import socket

computer_name = socket.gethostname()


if computer_name == 'oasis':
    datapath = '/home/pghosh/repos/datasets/celeba'
else:
    datapath = '/is/cluster/fast/pghosh/datasets/celebA/'

batch_size = 64


class VanillaVit:
    def __init__(self, epochs, batch_size, learning_rate):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate


class CelebA:
    def __init__(self):
        self.classes = 40
        self.img_size = 128
        self.vanilla_vit = VanillaVit(epochs=50, batch_size=64, learning_rate=0.001)


celeba_config = CelebA()