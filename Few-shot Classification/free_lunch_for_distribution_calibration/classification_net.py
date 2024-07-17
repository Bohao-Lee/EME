import torch
import torch.nn as nn

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()
        self.out = torch.nn.Linear(n_feature, n_output)
        
    def forward(self, input_feature):
        input_feature = self.out(input_feature)
        input_feature = torch.nn.functional.softmax(input_feature)
        return input_feature
    
    
class Net_2_layer(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net_2_layer, self).__init__()
        self.n_hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)
 
    def forward(self, x_layer):
        x_layer = torch.relu(self.n_hidden(x_layer))
        x_layer = self.out(x_layer)
        x_layer = torch.nn.functional.softmax(x_layer)
        return x_layer

    
class Net_3_layer(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net_3_layer, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_feature, n_feature//4, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(n_feature//4, n_feature//16, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(n_feature//16, n_output, bias=True),
        )
    
    def forward(self, input_feature):
        output_feature = self.network(input_feature)
        return output_feature
    

class Net_3_layer_relu(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net_3_layer_relu, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=n_feature, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=n_output),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        return self.main(input)
