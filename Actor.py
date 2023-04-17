import numpy as np
import torch
import torch.nn as nn

import torch.nn as nn

def init_weights_he_actor(layer: object) -> float:
    """Calculate the standard deviation of the distribution from which the initial weights will be sampled using He initializer.

    Args:
        layer (torch.nn.Linear): A linear layer in the actor network.

    Returns:
        std (float): A standard deviation of the normal distribution.
    """
    
    # Calculate the standard deviation for He initialization
    fan_in = layer.weight.data.size()[0]
    std = np.sqrt(2.0 / fan_in)
    return std

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=3):# , fc2_units=1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        
        # Set the random seed
        self.seed = torch.manual_seed(seed)
        
        # Define the neural network layers
        self.fc1  = nn.Linear(state_size, fc1_units)
        self.fc2  = nn.Linear(fc1_units,action_size)
      
        # Define activation functions
        self.relu = nn.ReLU()
        self.tanh=nn.Tanh()
     
        # Initialize the neural network weights
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize the weights of the neural network using He initialization."""
        self.fc1.weight.data.normal_(mean= 0 ,std=init_weights_he_actor(self.fc1))
        # self.fc2.weight.data.normal_(mean =0,std=init_weights_he_actor(self.fc2))
        self.fc2.weight.data.normal_(std=3e-3)
        
        
    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.relu(x)
        # Scale the actor output to the desired range
        # x = (x * (15 - 1)) + 1
        return x    