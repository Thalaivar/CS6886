from tqdm import tqdm
from network import *
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

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
    wandb.init(project='mini-assignment-3', dir='../')

    model = Network(in_channels=3)
    model.to(device)
    model.train()

    wandb.watch(model)

    # set these parameters
    batch_size = 64
    lr = 1e-1
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    n_epochs = 1000

    wandb.config.batch_size = batch_size
    wandb.config.loss_fn = type(criterion).__name__
    wandb.config.optimizer = type(optimizer).__name__
    wandb.config.lr = lr

    train_data = datasets.CIFAR100(root=data_dir, train=True, transform=transform_train, download=True)
    train_dataloader = DataLoader(train_data, batch_size, shuffle=True, num_workers=4)
    test_data = datasets.CIFAR100(root=data_dir, train=False, transform=transform_test, download=True)
    test_dataloader = DataLoader(test_data, batch_size, shuffle=True, num_workers=0)

    logging.info(f'Beginning training for {n_epochs} epochs ({n_epochs*len(train_dataloader)} steps) on device: {device}')
    
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

        scheduler.step()
        eval_acc = evaluate_model(model, test_dataloader)
        wandb.log({'eval accuracy': eval_acc}, step=steps)
        logging.info(f'epoch: {epoch} ; evaluation accuracy: {round(100 * eval_acc, 2)}%')    

    torch.save(model.parameters(), './final_model.wts')
    wandb.save("final_model.h5")

def evaluate_model(model, dataloader):
    model.eval()

    total, correct = 0, 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs).cpu()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.shape[0]
            correct += (predicted == labels.cpu()).sum().item()

    model.train()
    return correct / total

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train(data_dir='D:/IIT/Coursework/data')