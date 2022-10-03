import numpy as np
import torch
import torch.nn as nn

class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, 1024)
        self.ad_layer2 = nn.Linear(1024,1024)
        self.ad_layer3 = nn.Linear(1024, 1)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer1.bias.data.fill_(0.0)
        self.ad_layer2.bias.data.fill_(0.0)
        self.ad_layer3.bias.data.fill_(0.0)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.ad_layer3(x)
        x = self.sigmoid(x)
        return x

    def output_num(self):
        return 1
    
    def get_parameters(self):
        return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]
    
    
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

class AdversarialLayer(torch.autograd.Function):
    iter_num = 0
    alpha = 10
    low = 0.0
    high = 1.0
    max_iter = 10000.0
    @staticmethod
    def forward(ctx, input):
        AdversarialLayer.iter_num += 1
        output = input * 1.0
        return output
    
    @staticmethod
    def backward(ctx, gradOutput):
        coeff = calc_coeff(AdversarialLayer.iter_num,
                           high=AdversarialLayer.high, 
                           low=AdversarialLayer.low, 
                           alpha=AdversarialLayer.alpha,
                           max_iter=AdversarialLayer.max_iter)
        return -coeff * gradOutput


def PADA(iter_num, features, ad_net, grl_layer, weight_ad, use_gpu=True):
    ad_out = ad_net(grl_layer.apply(features))
#     ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()
    if use_gpu:
        dc_target = dc_target.cuda()
        weight_ad = weight_ad.cuda()
    return nn.BCELoss(weight=weight_ad.view(-1))(ad_out.view(-1), dc_target.view(-1))


def get_class_weight(loader, model, T=1.0):
    model.train(False)
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            _, outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    
    softmax_outputs = torch.nn.Softmax(dim=1)(all_output/T).detach()
    class_weight = torch.mean(softmax_outputs, dim=0)
#     class_weight = (class_weight / torch.mean(class_weight)).view(-1)
    class_weight = class_weight / torch.max(class_weight)
    class_weight = class_weight.view(-1)

    return class_weight