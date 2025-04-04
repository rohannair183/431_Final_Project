import numpy as np
import create_threshold_matrix as ct
from time import sleep
import pandas as pd
# Parameters
STATE_INCREMENT = 50
STATE_VALUES = [i * STATE_INCREMENT for i in range(37)]
transition_matrix = np.loadtxt(
    'markov_chain_transition_matrix_multiples_of_50.csv', delimiter=',', skiprows=1, usecols=range(1, 38))
NUM_DAYS = 10000
COST_HOLDING_PER_FT3 = 1.5  # Based on c = 10% of k
K1, K2 = 500, 750
k_per_m3 = 15
k_per_ft3 = k_per_m3 / 35.3147  # Conversion factor from m³ to ft³
fixed_3pl_cost = 800

np.random.seed(42)


def simulate_policy(threshold, logistics_mode, transition_matrix=transition_matrix):
    current_state_index = 0
    total_holding = 0
    total_shipping = 0
    num_shipments = 0

    for _ in range(NUM_DAYS):
        next_state_index = np.random.choice(
            range(len(STATE_VALUES)), p=transition_matrix[current_state_index]
        )
        current_state_index = next_state_index

        inventory = STATE_VALUES[current_state_index]
        total_holding += inventory * COST_HOLDING_PER_FT3

        if inventory >= threshold:
            # Shipping cost logic
            if logistics_mode == 'truck':
                if inventory <= 900:
                    total_shipping += K1
                else:
                    total_shipping += K2
            elif logistics_mode == '3pl':
                m3 = inventory * 0.0283168  # ft³ to m³
                total_shipping += fixed_3pl_cost + (k_per_m3 * m3)

            current_state_index = 0  # iInventory resets to 0
            num_shipments += 1
            continue

    avg_cost = (total_holding + total_shipping) / NUM_DAYS
    freq = NUM_DAYS / num_shipments if num_shipments > 0 else float('inf')
    return avg_cost, freq


# Try multiple thresholds
thresholds = list(range(300, 1801, 50))  # Reasonable steps
results = []

print("Running dynamic policy simulations...\n")

for threshold in thresholds:
    ct.create_transition_matrix(
        threshold, output_file=f'Threshold_Matrices/transition_matrix_{threshold}.csv')
    # Load the transition matrix for the current threshold
    threshold_matrix = np.loadtxt(
        f'Threshold_Matrices/transition_matrix_{threshold}.csv', delimiter=',', skiprows=1, usecols=range(1, 38))
    cost_truck, freq_truck = simulate_policy(
        threshold, 'truck', threshold_matrix)
    cost_3pl, freq_3pl = simulate_policy(threshold, '3pl', threshold_matrix)
    results.append({
        'Threshold': threshold,
        'Truck_Cost_per_Day': round(cost_truck, 2),
        'Truck_Frequency': round(freq_truck, 2),
        '3PL_Cost_per_Day': round(cost_3pl, 2),
        '3PL_Frequency': round(freq_3pl, 2),
    })

# Display results
df = pd.DataFrame(results)
df.to_csv('dynamic_simulation_results.csv', index=False)
print(df.to_string(index=False))

# Find optimal thresholds
min_truck = df.loc[df['Truck_Cost_per_Day'].idxmin()]
min_3pl = df.loc[df['3PL_Cost_per_Day'].idxmin()]
print("\nOptimal Threshold for Truck Rental:", min_truck['Threshold'], "→ Cost/Day:",
      min_truck['Truck_Cost_per_Day'], "→ Every", min_truck['Truck_Frequency'], "days")
print("Optimal Threshold for 3PL:", min_3pl['Threshold'], "→ Cost/Day:",
      min_3pl['3PL_Cost_per_Day'], "→ Every", min_3pl['3PL_Frequency'], "days")
