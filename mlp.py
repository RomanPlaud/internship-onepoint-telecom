## implement a simple pytorch MLP classifier

import torch 
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import f1_score

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class MLP2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
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

## define a function to test the model
def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad(): # disable gradient calculation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device) # move data to device
            output = model(data) # forward pass: compute predicted outputs by passing inputs to the model
            loss = criterion(output, target) # calculate the loss
            loss_test = nn.CrossEntropyLoss()(output, target)
            running_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    test_f1 = f1_score(target.cpu(), pred.cpu(), average='macro')
    return test_loss, test_acc, test_f1

## Define a dataset class for the data
class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, labels, hierarchy=None):
        self.X = X.toarray()
        self.labels = labels
    def __len__(self):
        return (self.X).shape[0]
    def __getitem__(self, index):
        # transform to tensor
        X = torch.from_numpy(self.X[index].astype('float32'))
        return X, self.labels[index]
    
## Define a function to get the data
def get_data(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    train_data = Dataset(X_train, y_train)
    val_data = Dataset(X_val, y_val)
    test_data = Dataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


