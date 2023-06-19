import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score


    
## define a function to train the model
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    ## iterate over the training data with tqdm
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) # move data to device
        optimizer.zero_grad() # clear the gradients of all optimized variables
        output = model(data) # forward pass: compute predicted outputs by passing inputs to the model
        loss = criterion(output, target) # calculate the loss
        loss.backward() # backward pass: compute gradient of the loss with respect to model parameters
        optimizer.step() # perform a single optimization step (parameter update)
        running_loss += loss.item() * data.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    return train_loss

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    preds = []
    targets = []
    with torch.no_grad(): # disable gradient calculation
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data) # forward pass: compute predicted outputs by passing inputs to the model
            loss = criterion(output, target) # calculate the loss
            running_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            preds.append(pred.cpu().numpy())
            targets.append(target.cpu().numpy())
    val_loss = running_loss / len(val_loader.dataset)
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    val_f1 = f1_score(targets, preds, average='macro')
    val_acc = correct / len(val_loader.dataset)
    return val_loss, val_acc, val_f1


## define a function to test the model
def test(model, test_loader, criterion, device, hierarchy, depth):
    model.eval()
    n = depth
    losses = np.zeros(n)
    correct = np.zeros(n)
    targets = [[] for _ in range(n)]
    preds = [[] for _ in range(n)]
    with torch.no_grad(): # disable gradient calculation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device) # move data to device
            output = model(data) # forward pass: compute predicted outputs by passing inputs to the model
            output = nn.Softmax(dim=1)(output)
            for i, hierarchy_level in enumerate(hierarchy):
                unique = torch.unique(hierarchy_level)
                output_level = torch.zeros(output.shape[0], (len(unique)), device=output.device)
                output_depth = torch.log(output_level.scatter_add(1, hierarchy_level.repeat(output.shape[0], 1), output))
                target_depth = hierarchy_level[target]
                loss_level = criterion(output_depth, target_depth)
                losses[i] += loss_level.item() * data.size(0)
                pred = output_depth.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct[i] += pred.eq(target_depth.view_as(pred)).sum().item()

                targets[i].extend(target_depth.cpu().numpy())
                preds[i].extend(pred.cpu().numpy())

    test_loss = losses / len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    test_f1 = np.zeros(depth)

    for i in range(n):
        test_f1[i] = f1_score(targets[i], preds[i], average='macro')
    return test_loss, test_acc, test_f1



def train(model, device, train_loader, val_loader, optimizer, criteria, epochs, scheduler):
    model.to(device)
    ## train the model
    best_loss = np.inf
    best_model = None
    for epoch in range(epochs):
        _ = train_one_epoch(model, train_loader, optimizer, criteria, device)
        test_loss, test_acc, test_f1 = validate(model, val_loader, criteria, device)
        if test_loss < best_loss:
            best_loss = test_loss
            best_model = model
        if epoch % 3 == 0:
            print(f'Epoch {epoch} - Val Loss: {test_loss:.4f} - Val Acc: {test_acc:.4f} - Val F1: {test_f1:.4f}')
        scheduler.step()
    return best_model