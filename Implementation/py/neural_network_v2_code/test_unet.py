import torch
from utils.NlosNet import Block
from utils.NlosNet import Encoder
from utils.NlosNet import Decoder
from utils.NlosNet import NlosNet


# Test the block
block = Block(1, 64)
x = torch.rand(1, 1, 572, 572)
print(f"Block test: {block(x).shape}")

# Test the encoder
enc = Encoder()
x = torch.rand(1, 3, 572, 572)
ftrs = enc(x)
print(f"Encoder test: {[f.shape for f in ftrs]}")

# Test the decoder
dec = Decoder()
x = torch.rand(1, 1024, 28, 28)
print(f"Decoder test: {dec(x, ftrs[::-1][1:]).shape}")

# Test the network
net = NlosNet()
x = torch.rand(1, 3, 572, 572)
print(net(x).shape)