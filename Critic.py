import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights_gl_critic(layer: object) -> tuple[float,float]:
    """Calculate the standard deviation of the distribution from which the initial weights will be sampled using glorot initializer.

    Args:
        layer (torch.nn.Linear): A linear layer in the actor network.

    Returns:
        Tuple of two float values. represent the lower and upper bounds of the uniform distribution from which the initial weights will be sampled.
    """
    
    # Calculate the standard deviation range for glorot initialization
    fan_in_input = layer.weight.data.size()[0]
    fan_in_output = layer.weight.data.size()[1]
    std = np.sqrt(2.0 / (fan_in_input + fan_in_output))
    return (-std, std)

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=3):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        
        # set the random seed 
        self.seed = torch.manual_seed(seed)
        
        # Define the neural network layers
        self.fcs1 = nn.Linear(state_size , fc1_units)
        self.fc2  = nn.Linear(fc1_units+action_size ,1)
       
        # Define activation function
        self.tanh = nn.Tanh()

        # Initialize the neural network weights
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the weights of the neural network using gloro initializer."""
        self.fcs1.weight.data.uniform_(*init_weights_gl_critic(self.fcs1))
        # self.fc2.weight.data.uniform_(*init_weights_gl_critic(self.fc2))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = self.fcs1(state)
        xs = self.tanh(xs)
        x = torch.cat([xs, action], dim=1)
        # x = self.fc1(x)
        # x = self.tanh(x)
        x = self.fc2(x)
        
        return x

