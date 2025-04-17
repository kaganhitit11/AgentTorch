from agent_torch.core.substep import SubstepObservation

class GetCurrentState(SubstepObservation):
    def forward(self, state):
        # Get relevant state variables
        agents = state['agents']['citizens']
        environment = state['environment']
        
        observation = {
            'opinion': agents['opinion'],
            'previous_opinion': agents['previous_opinion'],
            'household': agents['household'],
            'age': agents['age'],
            'gender': agents['gender'],
            'ethnicity': agents['ethnicity'],
            'global_adoption_rate': environment['global_adoption_rate'],
            'time': environment['time']
        }
        
        return observation