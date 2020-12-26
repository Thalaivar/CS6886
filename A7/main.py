import torch
import numpy as np

from torch.nn import Unfold
from winograd import Winograd
from fft_conv import fft_conv
from torch.autograd import Variable
from bonus import *

CPU_RANK_CP_DECOMP = 74
CPU_RANK_TUCKER = None

def direct_conv(input_tensor: torch.Tensor, filter: torch.Tensor) -> torch.Tensor:
    N, C1, H, W = input_tensor.shape
    M, C2, R, S = filter.shape

    assert C1 == C2, f'Input tensor channels ({C1}) do not match with filter channels ({C2})'
    C = C1

    output_tensor = Variable(torch.zeros(N, M, H - R + 1, W - S + 1, dtype=input_tensor.dtype), requires_grad=True)
    
    from tqdm import tqdm
    pbar = tqdm(total=N*C*(H - R + 1)*(W - S + 1)*M*R*S)

    for n in range(N):
        for h in range(H - R + 1):
            for w in range(W - S + 1):
                for m in range(M):
                    conv_out = 0.0
                    for c in range(C):
                        for r in range(R):
                            for s in range(S):
                                conv_out += filter[m, c, r, s] * input_tensor[n, c, h + r, w + s]
                                pbar.update(1)
                    output_tensor[n, m, h, w] = conv_out
    
    return output_tensor

def im2col(input_tensor: torch.Tensor, filter: torch.Tensor) -> torch.Tensor:
    N, C1, H, W = input_tensor.shape
    M, C2, R, S = filter.shape

    assert C1 == C2, f'Input tensor channels ({C1}) do not match with filter channels ({C2})'
    C = C1

    unfold = Unfold(kernel_size=(R, S))
    im_as_col = unfold(input_tensor)
    
    out_as_col = im_as_col.transpose(1, 2).matmul(filter.view(filter.size(0), -1).t()).transpose(1, 2)
    out = out_as_col.reshape(out_as_col.size(0), out_as_col.size(1), H - R + 1, W - S + 1).contiguous()
    return out
    
def winograd(input_tensor: torch.Tensor, filter: torch.Tensor) -> torch.Tensor:
    out = Winograd.forward(input_tensor, filter)
    return out

def fft(input_tensor: torch.Tensor, filter: torch.Tensor) -> torch.Tensor:
    N, C1, H, W = input_tensor.shape
    M, C2, R, S = filter.shape

    assert C1 == C2, f'Input tensor channels ({C1}) do not match with filter channels ({C2})'
    C = C1

    out = fft_conv(input_tensor, filter)
    return out

def cp_decomp(input_tensor: torch.Tensor, filter: torch.Tensor) -> torch.Tensor:
    # rank = determine_rank_for_cp_decomp(filter, rank_range=[20, 70], input_tensor=input_tensor) if rank is None else rank
    rank = max(filter.shape) // 3
    conv = cp_decomposition(filter, rank=rank)
    return conv(input_tensor)

def tucker_decomp(input_tensor: torch.Tensor, filter: torch.Tensor, determine_rank=False) -> torch.Tensor:
    if determine_rank:
        ranks = estimate_ranks(filter)
        print(f'Ranks for given filter: {ranks}')
    else:
        ranks = CPU_RANK_CP_TUCKER
    
    conv = tucker(filter, ranks)
    return conv(input_tensor)

def verify_output():
    from torch.nn.functional import conv2d

    N, C, H, W = 8, 3, 224, 224
    M, R, S = 32, 7, 7 

    weight = Variable(torch.rand(M, C, R, S, dtype=torch.float32), requires_grad=True)
    input_tensor = torch.rand(N, C, H, W, dtype=torch.float32)

    # out = im2col(input_tensor, weight)
    # out = direct_conv(input_tensor, weight)
    # out = winograd(input_tensor, weight)
    # out = fft(input_tensor, weight)
    # out = cp_decomp(input_tensor, weight, rank=CPU_RANK_CP_DECOMP)
    out = tucker_decomp(input_tensor, weight, determine_rank=True)

    print(f'Max discrepancy in results = {(conv2d(input_tensor, weight) - out).abs().max()}')

if __name__ == "__main__":
    verify_output()