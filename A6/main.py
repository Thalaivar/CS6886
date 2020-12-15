import wandb
import time
import torch
import torch.utils.checkpoint as checkpoint
import numpy as np

from fuseconv import FUSEConv
from bert import *
from torch.autograd import Variable

def checkpoint_FUSEConv():
    wandb.init(project="mini-assignment-6", name="delete")

    model = FUSEConv(in_channels=3, device='cuda:0')
    input_tensor = Variable(torch.rand(8, 3, 224, 224), requires_grad=True).to('cuda:0')

    model.to('cuda:0')
    model.train()

    out = model(input_tensor)
    mem_alloc = torch.cuda.max_memory_allocated() // (2**20)

    n_runs = 100
    start_t = time.perf_counter()
    for _ in range(n_runs):
        out = model(input_tensor)
        model.zero_grad()
        out.sum().backward()

    t_elapsed = time.perf_counter() - start_t
    t_elapsed /= n_runs

    print(f'\nMemory allocated: {mem_alloc} MB\nTime elapsed = {round(t_elapsed * 1000, 2)} ms')

    info = {'Memory allocated (MB)': mem_alloc, 'Time Taken': t_elapsed}
    wandb.config.update(info)
    wandb.save('./fuseconv.py')
    wandb.save('./main.py')

def checkpoint_bert():
    wandb.init(project="mini-assignment-6", name="experiment-2")

    input_ids = torch.LongTensor(np.random.randint(low=0, high=35, size=(24, 64))).to('cuda:0')
    input_mask = torch.LongTensor(np.random.randint(low=0, high=2, size=(24, 64))).to('cuda:0')
    token_type_ids = torch.LongTensor(np.random.randint(low=0, high=2, size=(24, 64))).to('cuda:0')

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertModel.from_pretrained('bert-large-uncased')

    model.to('cuda:0')
    model.train()

    _, out = model(input_ids, token_type_ids, input_mask)
    mem_alloc = torch.cuda.max_memory_allocated() // (2**20)

    n_runs = 30
    start_t = time.perf_counter()
    for _ in range(n_runs):
        _, out = model(input_ids, token_type_ids, input_mask)
        model.zero_grad()
        out.sum().backward()

    t_elapsed = time.perf_counter() - start_t
    t_elapsed /= n_runs

    print(f'\nMemory allocated: {mem_alloc} MB\nTime elapsed = {round(t_elapsed * 1000, 2)} ms')

    info = {'Memory allocated (MB)': mem_alloc, 'Time Taken': t_elapsed}
    wandb.config.update(info)
    wandb.save('./bert.py')
    wandb.save('./main.py')

def plot_compute_vs_memory(compute, memory):
    import seaborn as sn
    import matplotlib.pyplot as plt
    sn.set_theme()

    # plt.scatter(compute[0], memory[0], c='b', marker='o', label='No Checkpoint')
    plt.scatter(compute[1], memory[1], c='g', marker='v', label='Exp. 1')
    plt.scatter(compute[2], memory[2], c='r', marker='s', label='Exp. 2')
    plt.scatter(compute[3], memory[3], c='y', marker='p', label='Exp. 3')
    plt.scatter(compute[4], memory[4], c='k', marker='+', label='Exp. 4')
    # plt.scatter(compute[5], memory[5], c='m', marker='D', label='Exp. 5')
    plt.xlabel('Compute Time (ms)')
    plt.ylabel('Max Memory Allocated (MB)')
    plt.legend(loc='lower left')
    plt.show()

if __name__ == "__main__":
    # checkpoint_bert()

    # FUSECONV_MEM = [273, 44, 63, 63, 63, 44]
    # FUSECONV_COM = [52.79, 67.78, 67.37, 66.11, 64.79, 65.73]
    # plot_compute_vs_memory(FUSECONV_COM, FUSECONV_MEM)

    BERT_MEM = [5931, 1545, 1413, 4389, 1683]
    BERT_COM = [958.19, 1289.56, 1267.43, 1052.91, 1258.12]
    plot_compute_vs_memory(BERT_COM, BERT_MEM)
