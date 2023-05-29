import numpy as np
import torch

if __name__ == '__main__':
    file_path = "/DISK/qwt/models/DGR_weights/KITTI-v0.3-ResUNetBN2C-conv1-5-nout32.pth"
    checkpoint = torch.load(file_path)
    print(checkpoint)