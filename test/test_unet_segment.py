import torch
import src.unet_segment as segment
import numpy as np
from src.loss import DiceLoss

cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="mean")
dice_loss = DiceLoss()

downarm_channels = [16,16,16,16]
uparm_channels = [16,16,16,16]

net = segment.UnetSegment(1,16,downarm_channels, uparm_channels, 5)

s = 128
gt = torch.tensor(np.random.choice(range(5), [5,s,s,s]))

x = torch.rand([5,1,s,s,s])
y = net(x)

dice = dice_loss(y, gt)
cross_entropy = cross_entropy_loss(y, gt)

loss = dice + cross_entropy

loss.backward()