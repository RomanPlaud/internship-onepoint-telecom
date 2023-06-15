import argparse
import numpy as np
from data import get_data_preprocessed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import torch
from torch import nn
from mlp import MLP2, train, test, get_data
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from scipy.sparse import vstack

losses = {'cross_entropy': nn.CrossEntropyLoss()}

def parse_args():

    parser = argparse.ArgumentParser(description='Hierarchical classification')

    ## data
    parser.add_argument('--dataset', type=str, default='wikivitals')
    parser.add_argument('--method', type=str, default='bow')
    parser.add_argument('--depth', type=int, default=2)

    ## classifier
    parser.add_argument('--classifier', type=str, default='mlp')
    parser.add_argument('--loss', type=str, default='cross_entropy')    

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    ## load data
    if args.dataset == 'wikivitals':
        X_train, X_val, X_test, y_train, y_val, y_test, num_labels = get_data_preprocessed(dataset_name=args.dataset, method=args.method, depth=args.depth)

    ## train classifier
    if args.classifier == 'logistic regression':
        model = LogisticRegression(max_iter=500)
        ## concatenate train and validation
        X = np.vstack((X_train, X_val))
        model.fit(vstack([X_train, X_val]), np.concatenate([y_train, y_val], axis=0))
        y_pred = model.predict(X_test)
        test_f1 = f1_score(y_test, y_pred, average='macro')
        test_acc = np.mean(y_test==y_pred)
        print(f'Test Acc: {test_acc} - Test F1: {test_f1} - depth: {args.depth}')

    elif args.classifier == 'mlp':
        batch_size = 128
        epochs = 20
        lr = 0.001
        hidden_size = 128
        dropout = 0.5
        weight_decay = 0.0005
        gamma=0.9

        criteria = losses[args.loss]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_loader, val_loader, test_loader = get_data(X_train, y_train, X_val, y_val, X_test, y_test, batch_size)
        model = MLP2(X_train.shape[1], hidden_size, num_labels)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ExponentialLR(optimizer, gamma=gamma)

        model.to(device)

        ## train the model
        best_f1 = 0
        best_model = None
        for epoch in range(epochs):
            _ = train(model, train_loader, optimizer, criteria, device)
            test_loss, test_acc, test_f1 = test(model, val_loader, criteria, device)
            if test_f1 > best_f1:
                best_f1 = test_f1
                best_model = model
            print(f'Epoch {epoch} - Val Loss: {test_loss:.4f} - Val Acc: {test_acc:.4f} - Val F1: {test_f1:.4f}')
            scheduler.step()
        
        ## test the model
        test_loss, test_acc, test_f1 = test(best_model, test_loader, criteria, device)
        print(f'Test Acc: {test_acc:.4f} - Test F1: {test_f1:.4f} - depth: {args.depth}')



