import torch
import torch.nn as nn

EPSILON = 1e-20

class TorchLeakySoftmax(nn.Module):
    """
    leaky softmax, x_i = e^(x_i) / (sum_{k=1}^{n} e^(x_k) + coeff) where coeff >= 0

    usage::

        a = torch.zeros(3, 9)
        TorchLeakySoftmax().forward(a) # the output probability should be 0.1 over 9 classes

    """
    def __init__(self, coeff=1.0):
        super(TorchLeakySoftmax, self).__init__()
        self.coeff = coeff

        
    def forward(self, x):
        m = torch.max(x,axis=1, keepdim=True).values
        z = x - m
        numerator = torch.exp(z)
        denominator = torch.sum(numerator, dim=-1, keepdim=True)
        softmax = numerator/(denominator+self.coeff*torch.exp(-m))   
        return softmax, torch.clamp(torch.sum(softmax, dim=-1, keepdim=True), min=0.0, max=1.0)

    
def EntropyLoss(predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=EPSILON):
    """
    entropy for multi classification

    predict_prob should be size of [N, C]

    class_level_weight should be [1, C] or [N, C] or [C]

    instance_level_weight should be [N, 1] or [N]

    :param predict_prob:
    :param class_level_weight:
    :param instance_level_weight:
    :param epsilon:
    :return:
    """
    N, C = predict_prob.size()
    
    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'
        
    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    entropy = -predict_prob*torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * entropy * class_level_weight) / float(N)

class ClassifierAuxiliary(nn.Module):
    def __init__(self, in_feature, output_dim, dropout=0.5):
        super(ClassifierAuxiliary, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, output_dim),
            TorchLeakySoftmax(coeff=output_dim)
        )
        
    def forward(self, x):
        y = self.layers(x)
        return y

    def get_parameters(self):
        return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]