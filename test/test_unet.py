import torch
import src.unet_segment as segment
import numpy as np
import torch.nn.functional as func

downarm_channels = [4,8,16,32,64]
uparm_channels = [32,16,8,8,8]

net = segment.UnetSegment(1,16,downarm_channels, uparm_channels, 7)

gt = torch.tensor([[1,2,3],[4,5,6]])
out = func.one_hot(gt)

x = torch.rand([5,1,128,128,128])
y = net(x)