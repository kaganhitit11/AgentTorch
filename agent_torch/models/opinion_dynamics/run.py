import os
import torch
from util import create_opinion_dynamics_model

def run_simulation(config_path='config.yaml', output_dir='results'):
    # Create and initialize the model
    runner = create_opinion_dynamics_model(config_path)
    runner.init()
    
    # Run simulation
    num_episodes = runner.config['simulation_metadata']['num_episodes']
    num_steps = runner.config['simulation_metadata']['num_steps_per_episode']
    
    for episode in range(num_episodes):
        print(f"Running episode {episode + 1}/{num_episodes}")
        runner.step(num_steps)
        
        # Save results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            trajectory = {
                'global_adoption_rate': [],
                'opinions': [],
                'time': []
            }
            
            for state in runner.state_trajectory:
                trajectory['global_adoption_rate'].append(
                    state['environment']['global_adoption_rate'].item()
                )
                trajectory['opinions'].append(
                    state['agents']['citizens']['opinion'].clone()
                )
                trajectory['time'].append(
                    state['environment']['time'].item()
                )
            
            torch.save(trajectory, os.path.join(output_dir, f'episode_{episode}.pt'))
    
    return runner.state_trajectory

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Opinion Dynamics Simulation')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Directory to save results')
    
    args = parser.parse_args()
    
    run_simulation(args.config, args.output_dir)