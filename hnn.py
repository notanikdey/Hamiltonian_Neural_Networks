import numpy as np 
import torch 
import torch.nn as nn 

class HNN(nn.Module):
    #NN that learns vector fields that are conservative or solenoidal or their combination
    def __init__(self, input_dim, differentiable_model, field_type='solenoidal', baseline=False, assume_canonical_coords=True):
        super(HNN, self).__init__()

        self.baseline = baseline #if set to true, just returns raw outpus with no physics bias
        self.differentialble_model = differentiable_model
        self.assume_canonical_coords = assume_canonical_coords #whether J = standard symplectic matrix 
        #symplectic matrix J (when assume_canonical_coords is True and dim=2) or general antisymmetric one (for higher dim) 
        self.M = self.permutation_tensor(input_dim) 
        self.field_type = field_type


        def forward(self, x):
            if self.baseline:
                #if baseline=True, just a vanilla neural ODE and return raw output
                return self.differentiable_model(x) 
            y = self.differentiable_model(x)
            assert y.dim()==2 and y.shape[1] == 2, "Ouput should be [batch_size, 2]"
            return y.split(1,1)
        

        #Core HNN function 
        def time_derivative(self, x, t=None, separte_fields=False):
            #Takes in a batch of inputs x = (q,p) and returns the time derivative vector field z' 
            if self.baseline:
                return self.differentiable_model(x)
            
            F1, F2 = self.forward(x) #Split the model's output into two scalar potentials, F1(conservative field) & F2(solenoidal field)

            #Initializing borth vector fields as 0 
            conservative_field = torch.zeros_like(x)
            solenoidal_field = torch.zeros_like(x)
            
            #For conservative field
            if self.field_type != 'solenoidal':
                dF1 = torch.autograd.grad(F1.sum(), x, create_graph=True)[0]
                conservative_field = dF1 @ torch.eye(*self.M.shape)
            
            #For solenoidal field 
            
