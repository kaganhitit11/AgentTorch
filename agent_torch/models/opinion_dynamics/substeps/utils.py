import torch

def calculate_household_agreement(opinions, household_ids):
    """Calculate the level of opinion agreement within households"""
    unique_households = torch.unique(household_ids)
    agreement_scores = []
    
    for household in unique_households:
        mask = household_ids == household
        household_opinions = opinions[mask]
        
        if len(household_opinions) > 1:
            # Calculate standard deviation of opinions in household
            agreement = 1 - torch.std(household_opinions)
            agreement_scores.append(agreement.item())
    
    return torch.tensor(agreement_scores).mean()

def calculate_opinion_distribution(opinions, num_bins=10):
    """Calculate the distribution of opinions across the population"""
    hist = torch.histogram(opinions, bins=num_bins, range=(0, 1))
    return hist.hist / len(opinions)