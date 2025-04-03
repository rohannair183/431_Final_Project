import numpy as np
from scipy.stats import poisson
import pandas as pd
import itertools
from collections import defaultdict

def compute_volume_probabilities(num_orders):
    # Possible volumes per order
    volumes = [50, 100, 150]
    probs = [0.3, 0.5, 0.2]
    
    # Generate all combinations for num_orders
    combinations = list(itertools.product(volumes, repeat=num_orders))
    
    # Dictionary to store total volume probabilities
    vol_probs = defaultdict(float)
    
    # Calculate probability for each combination
    for combo in combinations:
        total_volume = sum(combo)
        # Probability: product of individual probs (independent orders)
        prob = 1.0
        for v in combo:
            prob *= probs[volumes.index(v)]
        vol_probs[total_volume] += prob
    
    # Convert to regular dict and return
    return dict(vol_probs)
def create_transition_matrix(threshold, output_file="transition_matrix.csv"):
    # Ensure threshold is a multiple of 50 and within 50-1800
    if threshold < 50 or threshold > 1800 or threshold % 50 != 0:
        raise ValueError("Threshold must be a multiple of 50 between 50 and 1800")
    
    # Define states: 0, 50, 100, ..., 1800
    max_state = 1800
    states = np.arange(0, max_state + 50, 50)
    n_states = len(states)  # Number of states
    
    # Poisson probabilities for 0 to 3 orders (lambda = 2)
    lambda_val = 2
    p_orders = poisson.pmf([0, 1, 2, 3], lambda_val)
    p_orders[3] = 1 - sum(p_orders[:3])  # Adjust P(N=3) for cap
    
    # Volume probabilities per order
    vol_1 = compute_volume_probabilities(1)
    # {50: 0.3, 100: 0.5, 150: 0.2}
    vol_2 = compute_volume_probabilities(2)
    # {
    #     100: 0.3**2 ,           # 50+50
    #     150: 2 * 0.3 * 0.5,     # 50+100, 100+50
    #     200: 0.5**2 + 2*0.3*0.2,            # 100 + 100, 150 + 50, 50 + 150
    #     250: 2*0.5*0.2,     # 100+150, 150+100
    #     300: 0.2**2             # 150+150
    # }
    vol_3 = compute_volume_probabilities(3)
    # {
    #     150: 0.3**3,                   # 50+50+50
    #     200: 3 * 0.3**2 * 0.5,         # 50+50+100, etc.
    #     250: 3 * 0.3**2 * 0.2 + 3 * 0.3 * 0.5**2,  # 50+50+150, 50+100+100
    #     300: 3 * 0.3 * 0.5 * 0.2 + 0.5**3,         # 50+100+150, 100+100+100
    #     350: 3 * 0.5**2 * 0.2 + 3*0.3*0.2**2,         # 100+100+150, 50+150+150
    #     400: 3 * 0.5 * 0.2**2,         # 100+150+150
    #     450: 0.2**3                    # 150+150+150
    # }
    
    # Initialize transition matrix
    P = np.zeros((n_states, n_states))
    
    # Fill the matrix
    for i in range(n_states):
        current_inventory = states[i]
        
        # If current inventory >= threshold, any order sends to 0
        if current_inventory >= threshold:
            P[i, 0] += 1  # P(N=1) + P(N=2) + P(N=3)
            continue
        # 0 orders: stay in same state
        P[i, i] = p_orders[0]
        # 1 order
        for v, prob in vol_1.items():
            next_inventory = current_inventory + v
            
            next_idx = int(next_inventory // 50)
            P[i, next_idx] += p_orders[1] * prob
    
        # 2 orders
        for v, prob in vol_2.items():
            next_inventory = current_inventory + v
            
            next_idx = int(next_inventory // 50)
            P[i, next_idx] += p_orders[2] * prob
        
        # 3 orders
        for v, prob in vol_3.items():
            next_inventory = current_inventory + v
            
            next_idx = int(next_inventory // 50)
            P[i, next_idx] += p_orders[3] * prob
    
    # Ensure rows sum to 1 (correct for rounding errors)
    row_sums = P.sum(axis=1)
    for i in range(n_states):
        if row_sums[i] > 0:
            P[i, :] /= row_sums[i]
    
    # Convert to DataFrame with state labels
    state_labels = [str(s) for s in states]
    df = pd.DataFrame(P, index=state_labels, columns=state_labels)
    
    # Save to CSV
    df.to_csv(output_file)
    print(f"Transition matrix saved to {output_file}")
    
    return P  # Return matrix for further use if needed

# Example usage
# threshold = 200
# matrix = create_transition_matrix(threshold, "transition_matrix_200.csv")

# # Test with a larger threshold
# threshold = 1800
# matrix = create_transition_matrix(threshold, "transition_matrix_1800.csv")



