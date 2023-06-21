import argparse

import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

from data import get_data_preprocessed
from mlp import MLP
from loss import HierarchicalBCELoss
from train import test, train



losses = {'cross_entropy': nn.CrossEntropyLoss, 'hierarchical_bce_loss': HierarchicalBCELoss}

def parse_args():

    parser = argparse.ArgumentParser(description='Hierarchical classification')

    ## data
    parser.add_argument('--dataset', type=str, default='wikivitals')
    parser.add_argument('--method', type=str, default='bow')
    parser.add_argument('--depth', type=int, default=2)

    ## classifier
    parser.add_argument('--classifier', type=str, default='mlp')
    parser.add_argument('--loss', type=str, default='cross_entropy')
    parser.add_argument('--weights', help='weights for hierarchical loss', default=None, type=float, nargs='+')  

    ## mlp
    parser.add_argument('--epochs', help='epochs', default=30, type=int)

    ## evaluation
    parser.add_argument('--kfold', default=False, type=bool)


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    ## load data
    if args.dataset == 'wikivitals':
        train_set, test_set, hierarchy = get_data_preprocessed(dataset_name=args.dataset, method=args.method, depth=args.depth)
        hierarchy = torch.from_numpy(hierarchy)

    ## train classifier
    if args.classifier == 'logistic regression':
        model = LogisticRegression(max_iter=500)
        ## concatenate train and validation
        model.fit(train_set.X, train_set.labels)
        y_pred = model.predict(test_set.X)
        test_f1 = f1_score(test_set.labels, y_pred, average='macro')
        test_acc = np.mean(test_set.labels==y_pred)
        print(f'Test Acc: {test_acc} - Test F1: {test_f1} - depth: {args.depth}')

    elif args.classifier == 'mlp':
        batch_size = 128
        epochs = args.epochs
        lr = 0.001
        hidden_size = 128
        dropout = 0.5
        weight_decay = 0.0005
        gamma=0.9

        if args.loss == 'hierarchical_bce_loss':
            criteria = losses[args.loss](args.weights, hierarchy)
        else:
            criteria = losses[args.loss]()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not args.kfold:

            train_set, val_set = torch.utils.data.random_split(train_set.dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

            model = MLP(train_set.dataset.X.shape[1], hidden_size, hierarchy.shape[1])
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = ExponentialLR(optimizer, gamma=gamma)

            best_model = train(model, device, train_loader, val_loader, optimizer, criteria, epochs, scheduler)
            
            ## test the model
            test_loss, test_acc, test_f1 = test(best_model, test_loader, nn.NLLLoss(), device, hierarchy, args.depth)

            for i in range(args.depth):
                print(f'Test Acc: {test_acc[i]:.4f} - Test F1: {test_f1[i]:.4f} - depth: {i}')

        elif args.kfold:
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_losses = []
            for fold, (train_indices, val_indices) in enumerate(kfold.split(train_set)):
                # Split the dataset into train and validation sets for this fold
                train_dataset = torch.utils.data.Subset(train_set, train_indices)
                val_dataset = torch.utils.data.Subset(train_set, val_indices)

                # Create dataloaders for training and validation
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                
                model = MLP(train_set.dataset.X.shape[1], hidden_size, hierarchy.shape[1])
                optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                scheduler = ExponentialLR(optimizer, gamma=gamma)

                best_model = train(model, device, train_loader, val_loader, optimizer, criteria, epochs, scheduler)

                ## test the model
                results = test(best_model, val_loader, nn.NLLLoss(), device, hierarchy, depth=args.depth)
                fold_losses.append(results)

            fold_losses = np.array(fold_losses)
            ## save the results in a .csv file with 95% confidence interval for each metric without erasing what is already in the file

            with open(f'./results/results.csv', 'a') as f:
                for i in range(args.depth):
                    f.write(f'{args.dataset},{args.method},{args.depth},{args.classifier},{args.loss},{args.weights},{args.depth -i}, {fold_losses[:,1, i].mean():.4f}, {1.96*fold_losses[:,1, i].std()/np.sqrt(5):.4f}, {fold_losses[:,2, i].mean():.4f}, {1.96*fold_losses[:,2, i].std()/np.sqrt(5):.4f}\n')
            