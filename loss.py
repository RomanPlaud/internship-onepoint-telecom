import torch
from torch import nn

class HierarchicalBCELoss:
    def __init__(self, weights, hierarchy, base_loss=nn.CrossEntropyLoss()):
        self.weights = weights
        self.hierarchy = hierarchy
        self.base_loss = base_loss

    def __call__(self, output, targets):
        loss = 0
        for hierarchy_level, weight in zip(self.hierarchy, self.weights):
            unique = torch.unique(hierarchy_level)
            output_level = torch.zeros(output.shape[0], (len(unique)), device=output.device)
            output_depth = output_level.scatter_add(1, hierarchy_level.repeat(output.shape[0], 1), output)
            target_depth = hierarchy_level[targets]
            loss += weight * self.base_loss(output_depth, target_depth)
        return loss