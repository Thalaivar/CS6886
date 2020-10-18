from tqdm import tqdm
from network import *
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

import os
import time
import logging
import wandb
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([                                                                                                                                                  
                        transforms.RandomCrop(32, padding=4),                                                                                                                               
                        transforms.RandomHorizontalFlip(),                                                                                                                                  
                        transforms.ToTensor(),                                                                                                                                              
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),                                                                                           
                    ])              

transform_test = transforms.Compose([                                                                                                                                                   
                        transforms.ToTensor(),                                                                                                                                              
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),                                                                                           
                    ])

def train(data_dir: str):
    update_freq = 100
    wandb.init(project='mini-assignment-3', dir='../data/')
    checkpoint_dir = data_dir + '/' + wandb.run.name
    os.mkdir(checkpoint_dir)

    model = Network(in_channels=3)
    model.to(device)
    model.train()

    wandb.watch(model)

    # set these parameters
    batch_size = 64
    lr = 5e-2
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-5)
    n_epochs = 1000 
    
    train_data = datasets.CIFAR100(root=data_dir, train=True, transform=transform_train, download=True)
    train_dataloader = DataLoader(train_data, batch_size, shuffle=True, num_workers=4)
    test_data = datasets.CIFAR100(root=data_dir, train=False, transform=transform_test, download=True)
    test_dataloader = DataLoader(test_data, batch_size, shuffle=True, num_workers=0)

    scheduler = None
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader)*n_epochs, eta_min=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, threshold=1e-2, min_lr=1e-6, factor=0.5)

    wandb.config.batch_size = batch_size
    wandb.config.loss_fn = type(criterion).__name__
    wandb.config.optimizer = type(optimizer).__name__
    wandb.config.lr = lr
    if scheduler is not None:
        wandb.config.scheduler = type(scheduler).__name__

    logging.info(f'Beginning training for {n_epochs} epochs ({n_epochs*len(train_dataloader)} steps) on device: {device}')
    
    start_time = time.time()
    steps = 0
    running_loss = 0.0
    for epoch in range(n_epochs):
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            steps += 1

            if steps % update_freq == 0:
                wandb.log({'loss': running_loss/update_freq}, step=steps)
                # logging.info(f'Steps: {steps} ; Loss: {running_loss/update_freq}')
                running_loss = 0.0

            # if scheduler is not None:
            #     scheduler.step()
                
        eval_acc, eval_loss = evaluate_model(model, test_dataloader)
        wandb.log({'eval accuracy': eval_acc, 'eval loss': eval_loss}, step=steps)
        logging.info(f'epoch: {epoch} ; evaluation accuracy: {round(100 * eval_acc, 2)}%; [{round(time.time() - start_time, 2)} secs]') 
        if epoch % 50 == 0:
            checkpoint(model, epoch, start_time, checkpoint_dir)  
        scheduler.step(eval_acc)

    t_elapsed = time.time() - start_time
    wandb.config.time = t_elapsed
    logging.info(f'completed {n_epochs} epochs of training in {t_elapsed} secs')
    torch.save(model.state_dict(), './final_model.wts')
    wandb.save("final_model.h5")

def evaluate_model(model, dataloader):
    model.eval()

    loss_fn = nn.CrossEntropyLoss()
    total, correct, running_loss = 0, 0, 0.0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs).cpu()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.shape[0]
            correct += (predicted == labels.cpu()).sum().item()

            loss = loss_fn(outputs, labels.cpu())
            running_loss += loss.item()
    model.train()
    return correct / total, running_loss/len(dataloader)

def checkpoint(model, epoch, start_time, data_dir):
    save_file = data_dir + f'/ep_{epoch}_{time.time() - start_time}.wts'
    torch.save(model.state_dict(), save_file)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train(data_dir='D:/IIT/Coursework/data')