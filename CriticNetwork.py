import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from einops import rearrange, repeat

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)
    
class Swish_(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

SiLU = nn.SiLU if hasattr(nn, 'SiLU') else Swish_

class ValueNet(torch.nn.Module):
    # simple linear network
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        # self.embedding = nn.Embedding(state_dim, 50)
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

        print("------use_orthogonal_init------")
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2, gain=0.01)

    def forward(self, mlp_input):
        if len(mlp_input.shape) == 3:
            x1 = rearrange(mlp_input, 'a b c ->c (a b)')   # state_dim, 3, embedding_size -> embedding_size, 3, state_dim
        else:
            x1 = rearrange(mlp_input, 'l a b c ->l c (a b)')   # state_dim, 3, embedding_size -> embedding_size, 3, state_dim
        x2 = F.relu(self.fc1(x1))

        return torch.mean(self.fc2(x2), dim=1)


class PaiNNValueNet(nn.Module):
    """PaiNN style update network. Models the interaction between scalar and vectorial part"""

    def __init__(self, r_state_dim, r_hidden_dim, node_size):
        super().__init__()

        self.pre_linear = nn.Linear(node_size, node_size, bias=False)

        self.combined_mlp = nn.Sequential(
            nn.Linear(2 * node_size, node_size),
            nn.SiLU(),
            nn.Linear(node_size, 3 * node_size),
        )
        
        self.deep_to_rein_mlp = nn.Sequential(
            nn.Linear(r_state_dim, r_hidden_dim),
            nn.SiLU(),
            nn.Linear(r_hidden_dim, 1),
        )


    def forward(self, node_state_scalar, node_state_vector):
        """
        Args:
            node_state_scalar (tensor): Node states (num_nodes, node_size)
            node_state_vector (tensor): Node states (num_nodes, 3, node_size)

        Returns:
            Tuple of 2 tensors:
                updated_node_state_scalar (num_nodes, node_size)
                updated_node_state_vector (num_nodes, 3, node_size)
        """
        '''if len(node_state_scalar.shape) == 2 and len(node_state_vector.shape) == 3:
            node_state_scalar = rearrange(node_state_scalar, 'a b -> b a') # num_nodes,embedding_size -----> embedding_size, num_nodes
            node_state_vector = rearrange(node_state_vector, 'a b c -> c b a')  # num_nodes, 3, embedding_size -----> embedding_size, 3, num_nodes

        elif len(node_state_scalar.shape) == 3 and len(node_state_vector.shape) == 4:
            node_state_scalar = rearrange(node_state_scalar, 'a b c-> a c b') # len(transition_dict), num_nodes,embedding_size -----> len(transition_dict), embedding_size, num_nodes
            node_state_vector= rearrange(node_state_vector, 'd a b c -> d c b a')  # len(transition_dict), num_nodes, 3, embedding_size -----> len(transition_dict), embedding_size, 3, num_nodes
        '''
        pre_linear = self.pre_linear(node_state_vector)  # len(transition_dict), num_nodes, 3, node_size
        

        if len(node_state_scalar.shape) == 2:
            pre_norm = torch.linalg.norm(pre_linear, dim=1, keepdim=False)  # len(transition_dict), num_nodes, node_size
            mlp_input = torch.cat(
                (node_state_scalar, pre_norm), dim=1
            )  # num_nodes, node_size*2
            
            mlp_output = self.combined_mlp(mlp_input)
            mlp_output = rearrange(node_state_scalar, 'a b -> b a') # num_nodes, node_size * 3 -> node_size*3, num_nodes
        if len(node_state_scalar.shape) == 3:
            pre_norm = torch.linalg.norm(pre_linear, dim=2, keepdim=False)  # len(transition_dict), num_nodes, node_size
        
            mlp_input = torch.cat(
                (node_state_scalar, pre_norm), dim=2
            )  # len(transition_dict), num_nodes, node_size*2
            
            mlp_output = self.combined_mlp(mlp_input)
            mlp_output = rearrange(node_state_scalar, 'a b c ->a c b') # len(transition_dict), num_nodes, node_size * 3 -> len(transition_dict), node_size*3, num_nodes
            
            # print(torch.mean(self.deep_to_rein_mlp(mlp_output), dim = 1))

        return torch.mean(self.deep_to_rein_mlp(mlp_output), dim = 1)