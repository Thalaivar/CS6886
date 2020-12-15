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

    _, out = model(input_ids, token_type_ids, input_mask, output_all_encoded_layers=False)
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

    plt.scatter(compute[0], memory[0], 'bo', label='No Checkpoint')
    plt.scatter(compute[1], memory[1], 'gv', label='Exp. 1')
    plt.scatter(compute[2], memory[2], 'rs', label='Exp. 2')
    plt.scatter(compute[3], memory[3], 'yp', label='Exp. 3')
    plt.scatter(compute[4], memory[4], 'k+', label='Exp. 4')
    plt.scatter(compute[5], memory[5], 'mD', label='Exp. 5')
    plt.xlabel('Compute Time (ms)')
    plt.ylabel('Max Memory Allocated (MB)')
    plt.show()

if __name__ == "__main__":
    checkpoint_bert()