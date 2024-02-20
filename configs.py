import socket

computer_name = socket.gethostname()


if computer_name == 'oasis':
    datapath = '/home/pghosh/repos/datasets/celeba'
else:
    raise ValueError("We don't recognize this PC, we don't know where to put your dataset")
