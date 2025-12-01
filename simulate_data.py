"""
Data Simulation for Shapley Attribution

This script generates simulated marketing attribution data with different scenarios:
- Complete data: All coalitions available with sufficient sample sizes
- Missing data: Some coalitions are missing
- Small sample size: Some coalitions have insufficient sample sizes
"""

import pandas as pd
import numpy as np
from itertools import combinations, chain


def powerset(iterable):
    """
    Generate all possible subsets (coalitions) of an iterable.
    
    Args:
        iterable: List of items
        
    Yields:
        tuple: Each possible subset
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def generate_data(channels, scenario='complete', min_count=5, 
                 base_rate=0.05, lift_per_channel=0.02, seed=42):
    """
    Generate simulated attribution data.
    
    Args:
        channels (list): List of channel names
        scenario (str): Scenario type:
            - 'complete': All coalitions available with sufficient sample sizes
            - 'missing': Some coalitions are missing (random 30% missing)
            - 'small_sample': Some coalitions have small sample sizes (< min_count)
            - 'mixed': Combination of missing and small sample sizes
        min_count (int): Minimum sample size threshold
        base_rate (float): Base conversion rate for empty coalition
        lift_per_channel (float): Additional lift per channel in coalition
        seed (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: DataFrame with columns for each channel (0/1), 
                     'sample_size', and 'conversion_rate'
    """
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    
    # Generate all possible coalitions
    all_combos = list(powerset(channels))
    
    # Calculate conversion rates for each coalition
    def conv_prob(combo):
        """Calculate conversion probability for a coalition."""
        return min(1.0, base_rate + lift_per_channel * len(combo))
    
    rows = []
    
    for combo in all_combos:
        combo_set = set(combo)
        rate = conv_prob(combo)
        
        # Determine sample size based on scenario
        if scenario == 'complete':
            # All have sufficient sample sizes
            size = random_state.randint(50, 200)
        
        elif scenario == 'missing':
            # Randomly drop 30% of coalitions (except empty and full)
            if len(combo) > 0 and len(combo) < len(channels):
                if random_state.random() < 0.3:
                    continue  # Skip this coalition
            size = random_state.randint(50, 200)
        
        elif scenario == 'small_sample':
            # Some coalitions have small sample sizes
            if len(combo) > 1 and len(combo) < len(channels):
                # 40% chance of small sample size
                if random_state.random() < 0.4:
                    size = random_state.randint(1, min_count - 1)
                else:
                    size = random_state.randint(50, 200)
            else:
                size = random_state.randint(50, 200)
        
        elif scenario == 'mixed':
            # Combination: some missing, some small sample
            if len(combo) > 0 and len(combo) < len(channels):
                rand = random_state.random()
                if rand < 0.2:
                    continue  # Missing
                elif rand < 0.5:
                    size = random_state.randint(1, min_count - 1)  # Small sample
                else:
                    size = random_state.randint(50, 200)
            else:
                size = random_state.randint(50, 200)
        
        else:
            raise ValueError(f"Unknown scenario: {scenario}. Use 'complete', 'missing', 'small_sample', or 'mixed'")
        
        # Create row with one-hot encoding
        row = {ch: int(ch in combo_set) for ch in channels}
        row['sample_size'] = size
        row['rate'] = rate
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df.sort_values(by=channels).reset_index(drop=True)


def generate_custom_data(channels, missing_combos=None, small_sample_combos=None,
                        base_rate=0.05, lift_per_channel=0.02, 
                        default_size=(50, 200), small_size=(1, 4), seed=42):
    """
    Generate data with custom missing and small sample size coalitions.
    
    Args:
        channels (list): List of channel names
        missing_combos (list): List of channel combinations to exclude (as tuples or sets)
        small_sample_combos (list): List of channel combinations with small sample sizes
        base_rate (float): Base conversion rate
        lift_per_channel (float): Additional lift per channel
        default_size (tuple): (min, max) for default sample sizes
        small_size (tuple): (min, max) for small sample sizes
        seed (int): Random seed
        
    Returns:
        pd.DataFrame: Generated data
    """
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    
    # Convert missing_combos to sets for easy lookup
    missing_sets = set()
    if missing_combos:
        for combo in missing_combos:
            missing_sets.add(frozenset(combo))
    
    small_sample_sets = set()
    if small_sample_combos:
        for combo in small_sample_combos:
            small_sample_sets.add(frozenset(combo))
    
    all_combos = list(powerset(channels))
    rows = []
    
    for combo in all_combos:
        combo_set = frozenset(combo)
        
        # Skip if in missing list
        if combo_set in missing_sets:
            continue
        
        rate = min(1.0, base_rate + lift_per_channel * len(combo))
        
        # Determine sample size
        if combo_set in small_sample_sets:
            size = random_state.randint(small_size[0], small_size[1])
        else:
            size = random_state.randint(default_size[0], default_size[1])
        
        row = {ch: int(ch in combo) for ch in channels}
        row['sample_size'] = size
        row['rate'] = rate
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df.sort_values(by=channels).reset_index(drop=True)


if __name__ == "__main__":
    # Example 1: Complete data
    print("Generating complete data scenario...")
    channels = ['A', 'B', 'C', 'D']
    df_complete = generate_data(channels, scenario='complete')
    df_complete.to_csv("simulated_data_complete.csv", index=False)
    print(f"Generated {len(df_complete)} rows")
    
    # Example 2: Missing data
    print("\nGenerating missing data scenario...")
    df_missing = generate_data(channels, scenario='missing')
    df_missing.to_csv("simulated_data_missing.csv", index=False)
    print(f"Generated {len(df_missing)} rows")
    
    # Example 3: Small sample sizes
    print("\nGenerating small sample size scenario...")
    df_small = generate_data(channels, scenario='small_sample', min_count=5)
    df_small.to_csv("simulated_data_small.csv", index=False)
    print(f"Generated {len(df_small)} rows")
    
    # Example 4: Mixed scenario
    print("\nGenerating mixed scenario...")
    df_mixed = generate_data(channels, scenario='mixed', min_count=5)
    df_mixed.to_csv("simulated_data_mixed.csv", index=False)
    print(f"Generated {len(df_mixed)} rows")
    
    # Example 5: Custom scenario
    print("\nGenerating custom scenario...")
    df_custom = generate_custom_data(
        channels,
        missing_combos=[('A', 'B'), ('C', 'D')],
        small_sample_combos=[('A', 'C'), ('B', 'D')]
    )
    df_custom.to_csv("simulated_data_custom.csv", index=False)
    print(f"Generated {len(df_custom)} rows")
    
    print("\nAll data files generated successfully!")
