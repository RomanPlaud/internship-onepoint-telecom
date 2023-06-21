import torch
from torch import nn

class HierarchicalBCELoss:
    def __init__(self, weights, hierarchy, base_loss=nn.NLLLoss(), eps=1e-10):
        self.weights = weights
        self.hierarchy = hierarchy
        self.base_loss = base_loss
        self.eps = eps

    def __call__(self, output, targets):
        loss = 0
        output = nn.Softmax(dim=1)(output)
        for hierarchy_level, weight in zip(self.hierarchy, self.weights):
            unique = torch.unique(hierarchy_level)
            output_level = torch.zeros(output.shape[0], (len(unique)), device=output.device)
            output_depth = torch.log(output_level.scatter_add(1, hierarchy_level.repeat(output.shape[0], 1), output) + self.eps)
            target_depth = hierarchy_level[targets]
            loss_level = self.base_loss(output_depth, target_depth)
            loss += weight * loss_level
        return loss