import torch 
import torch.nn as nn 


class BaselineMLP(nn.Module):
    def __init__(self):
       super().__init__()
       self.layer_1 = nn.Linear(in_features=2, out_features=200)
       self.layer_2 = nn.Linear(in_features=200, out_features=200)
       self.layer_3 = nn.Linear(in_features=200, out_features=2)

       for l in [self.layer_1, self.layer_2, self.layer_3]:
           nn.init.orthogonal_(l.weight)
       
    def forward(self, x):
           h = self.tanh(self.layer_1(x))
           h = self.tanh(self.layer_2(h))

           return self.layer_3(h)
       

           


        