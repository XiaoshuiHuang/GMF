# coding=utf-8
import torch
import numpy as np

if __name__ == '__main__':

    # checkpoint = "/home/qwt/code/DeepGlobalRegistration-master-test-modify/outputs/checkpoint_4_0.7575.pth"
    # state = torch.load(checkpoint)
    data = torch.as_tensor([1,2])
    data = (data)
    data = torch.as_tensor(data)
    # data= torch.cat(data,dim=0)
    print(data.shape)
    pass