import os
import sys
import torch
from agent_torch.core import Runner, Registry
from agent_torch.core.substep import SubstepObservation, SubstepAction, SubstepTransition
from agent_torch.core.helpers import get_by_path, read_config, read_from_file

# Helper functions
def get_var(state, var):
    """
    Retrieves a value from the current state of the model.
    """
    return get_by_path(state, re.split('/', var))

# Register helper functions
@Registry.register_helper('random_float', 'initialization')
def random_float(shape, params):
    """
    Generates initial opinions between 0 and 1
    """
    return torch.rand(shape)

@Registry.register_helper('household_network', 'network')
def household_network(params):
    """
    Creates network connections based on household relationships
    """
    population = params.get('population')
    household_ids = population['household'].unique()
    
    # Create adjacency matrix based on household relationships
    n_agents = len(population)
    adj_matrix = torch.zeros((n_agents, n_agents))
    
    for household in household_ids:
        members = population[population['household'] == household].index
        for i in members:
            for j in members:
                if i != j:
                    adj_matrix[i][j] = 1
                    
    return adj_matrix

# Main model initialization
def create_opinion_dynamics_model(config_path='config.yaml'):
    """
    Creates and initializes the opinion dynamics model
    """
    config = read_config(config_path)
    
    # Initialize registry
    registry = Registry()
    registry.register(read_from_file, 'read_from_file', 'initialization')
    registry.register(random_float, 'random_float', 'initialization')
    registry.register(household_network, 'household_network', 'network')
    
    # Create runner
    runner = Runner(config, registry)
    
    return runner