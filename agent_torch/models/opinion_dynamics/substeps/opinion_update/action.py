from agent_torch.core.substep import SubstepAction
import torch

class UpdateOpinions(SubstepAction):
    def forward(self, observation):
        # Compute opinion updates
        opinions = observation['opinion']
        household_network = observation['network']['household']
        
        # Calculate household influence
        household_influence = torch.matmul(household_network, opinions)
        household_degrees = household_network.sum(dim=1, keepdim=True)
        household_influence = household_influence / (household_degrees + 1e-8)
        
        # Update opinions
        new_opinions = 0.7 * opinions + 0.3 * household_influence
        
        return {'new_opinions': new_opinions}