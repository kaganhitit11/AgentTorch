from agent_torch.core.substep import SubstepTransition

class ApplyOpinionUpdates(SubstepTransition):
    def forward(self, state, action):
        # Update state with new opinions
        state['agents']['citizens']['previous_opinion'] = state['agents']['citizens']['opinion']
        state['agents']['citizens']['opinion'] = action['new_opinions']
        
        # Update global adoption rate
        adoption_threshold = 0.8
        adopters = (state['agents']['citizens']['opinion'] > adoption_threshold).float()
        state['environment']['global_adoption_rate'] = adopters.mean()
        
        return state