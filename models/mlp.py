
import torch
from torch.nn import Module
from torch.nn import Conv1d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import LogSoftmax, LogSigmoid
from torch import flatten


class MLP(Module):

    def __init__(self, numChannels, classes, multilabel=False):
        # call the parent constructor
        super(MLP, self).__init__()

        self.name = "MLP"

        self.fc1 = Linear(in_features=numChannels, out_features=256)
        #self.fc2 = Linear(in_features=1024, out_features=256)
        self.fc3 = Linear(in_features=256, out_features=classes)
        self.relu = ReLU()

        if multilabel:
            self.activation = LogSigmoid()
        else:
            self.activation = LogSoftmax(dim=1)
        return

    def forward(self, x):

        # x: BS x 1 x C

        #print("input", x.shape)

        x = x.squeeze(-1) # pooling temporal dim (1)

        x = self.fc1(x)
        x = self.relu(x)

        #x = self.fc2(x)
        #x = self.relu(x)

        logits = self.fc3(x)

        output = self.activation(logits)

        # return the output predictions
        return logits



def generate_model(num_classes, n_input_channels, multilabel=False):
    model = MLP(n_input_channels, num_classes, multilabel)
    print("Model", model)
    return model

