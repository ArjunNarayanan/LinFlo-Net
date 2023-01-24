import torch
import src.unet_segment as segment
import numpy as np
import torch.nn.functional as func

cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
downarm_channels = [16,16,16,16]
uparm_channels = [16,16,16,16]

net = segment.UnetSegment(1,16,downarm_channels, uparm_channels, 5)

s = 32
gt = torch.tensor(np.random.choice(range(5), [5,s,s,s]))

x = torch.rand([5,1,s,s,s])
y = net(x)

loss = cross_entropy_loss(y, gt)
loss = loss[gt > 0].mean()
