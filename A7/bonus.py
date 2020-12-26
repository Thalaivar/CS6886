import VBMF
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker


def cp_decomposition(W: torch.Tensor, rank) -> nn.Module:
    last, first, vertical, horizontal = parafac(W.cpu().detach().numpy(), rank=rank, init='svd')[1]
    
    conv1 = nn.Conv2d(in_channels=first.shape[0], out_channels=rank, kernel_size=1, bias=False)
    conv2 = nn.Conv2d(in_channels=rank, out_channels=rank, kernel_size=(vertical.shape[0], 1), groups=rank, bias=False)
    conv3 = nn.Conv2d(in_channels=rank, out_channels=rank, kernel_size=(1, horizontal.shape[0]), groups=rank, bias=False)
    conv4 = nn.Conv2d(in_channels=rank, out_channels=last.shape[0], kernel_size=1, bias=False)

    conv1.weight.data = torch.from_numpy(np.transpose(first)).unsqueeze(-1).unsqueeze(-1)
    conv2.weight.data = torch.from_numpy(np.transpose(vertical)).unsqueeze(1).unsqueeze(-1)
    conv3.weight.data = torch.from_numpy(np.transpose(horizontal)).unsqueeze(1).unsqueeze(1)
    conv4.weight.data = torch.from_numpy(last).unsqueeze(-1).unsqueeze(-1)

    return nn.Sequential(conv1, conv2, conv3, conv4)

def determine_rank_for_cp_decomp(W: torch.Tensor, rank_range: int, input_tensor: torch.Tensor):
    from tqdm import tqdm
    from torch.nn.functional import conv2d

    min_delta = np.infty
    for rank in tqdm(range(*rank_range)):
        decomp_conv = cp_decomposition(W, rank)
        delta = (conv2d(input_tensor, W) - decomp_conv(input_tensor)).abs().max()
        if delta < min_delta:
            min_rank = rank
    
    print(f'Minimum rank required to maintain accuracy: {min_rank}')
    return min_rank

def estimate_ranks(W: torch.Tensor):
    W = W.detach().numpy()
    unfold_0 = tl.base.unfold(W, 0) 
    unfold_1 = tl.base.unfold(W, 1)
    _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
    _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
    ranks = [diag_0.shape[0], diag_1.shape[1]]
    return ranks

def tucker(W: torch.Tensor, ranks):
    core, [last, first] = partial_tucker(W.detach().numpy(), modes=[0, 1], rank=ranks, init='svd')   
    kernel_size = tuple(W.shape[2:])

    conv1 = nn.Conv2d(in_channels=first.shape[0], out_channels=first.shape[1], kernel_size=1, bias=False)
    conv2 = nn.Conv2d(in_channels=core.shape[1], out_channels=core.shape[0], kernel_size=kernel_size, bias=False)
    conv3 = nn.Conv2d(in_channels=last.shape[1], out_channels=last.shape[0], kernel_size=1, bias=False)
    
    conv1.weight.data = torch.from_numpy(np.transpose(first)).unsqueeze(-1).unsqueeze(-1)
    conv2.weight.data = torch.from_numpy(core)
    conv3.weight.data = torch.from_numpy(last).unsqueeze(-1).unsqueeze(-1)

    return nn.Sequential(conv1, conv2, conv3)