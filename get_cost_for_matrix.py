import numpy as np
import pandas as pd
from scipy.stats import poisson

def compute_steady_state_from_csv(csv_file, output_file="steady_state_probabilities.csv"):
    # Read the transition matrix from CSV
    df = pd.read_csv(csv_file, index_col=0)  # First column is index (states)
    P = df.values  # Convert to NumPy array
    
    # Verify matrix shape
    if P.shape != (37, 37):
        raise ValueError(f"Expected a 37x37 matrix, got {P.shape}")
    
    # Check row sums (should be ~1 for a valid transition matrix)
    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1, atol=1e-5):
        print("Warning: Some row sums deviate from 1:", row_sums)
    
    # Solve for steady-state: pi * P = pi, sum(pi) = 1
    n = P.shape[0]
    A = P.T - np.eye(n)  # P^T - I
    A[-1] = np.ones(n)   # Replace last row with sum-to-1 constraint
    b = np.zeros(n)
    b[-1] = 1            # Right-hand side: [0, 0, ..., 1]
    
    # Solve the system
    pi = np.linalg.solve(A, b)
    
    # Ensure non-negativity and normalize
    pi = np.clip(pi, 0, None)
    pi /= pi.sum()  # Re-normalize to ensure sum = 1
    
    # Create output with state labels
    states = np.arange(0, 1850, 50)  # 0 to 1800, 37 states
    steady_state = pd.Series(pi, index=states, name="Probability")
    
    # Print result
    print("Steady-State Probabilities:")
    print(steady_state)
    print("Sum of probabilities:", steady_state.sum())
    
    # Save to CSV
    steady_state.to_csv(output_file, header=True)
    print(f"Steady-state probabilities saved to {output_file}")
    
    return pi

import numpy as np
import pandas as pd

def calculate_holding_cost(steady_state_csv, threshold, holding_cost_per_unit):
    # Read the steady-state probabilities from CSV
    df = pd.read_csv(steady_state_csv, index_col=0)  # First column is state index (0, 50, ..., 1800)
    pi = df["Probability"].values  # Extract probability column
    
    # Verify length
    if len(pi) != 37:
        raise ValueError(f"Expected 37 states, got {len(pi)}")
    
    # Define states (inventory levels)
    states = np.arange(0, 1850, 50)  # 0, 50, 100, ..., 1800
    
    # Calculate holding cost: pi * state * holding_cost_per_unit
    holding_costs = pi * states * holding_cost_per_unit
    total_holding_cost_per_day = np.sum(holding_costs)
    p_orders = poisson.pmf([0, 1, 2, 3], 2)
    p_orders[3] = 1 - sum(p_orders[:3])  # Adjust P(N=3) for cap
    orders_per_day = (1 * p_orders[1] + 2 * p_orders[2] + 3 * p_orders[3])
    volume_per_order = 0.3 * 50 + 0.5 * 100 + 0.2 * 150
    cycle_length = threshold / (orders_per_day * volume_per_order)
    total_holding_cost = total_holding_cost_per_day * cycle_length

    # print(f"\nTotal Expected Holding Cost per Day: {total_holding_cost:.6f}")
    return total_holding_cost

def compute_truck_cost(steady_state_csv, threshold, truck_1_cost, truck_2_cost, truck_1_capacity=900, truck_2_capacity=1800):
    # Read the steady-state probabilities from CSV
    df = pd.read_csv(steady_state_csv, index_col=0)  # First column is state index (0, 50, ..., 1800)
    pi = df["Probability"].values  # Extract probability column
    
    # Verify length
    if len(pi) != 37:
        raise ValueError(f"Expected 37 states, got {len(pi)}")
    
    # Define states (inventory levels)
    states = np.arange(0, 1850, 50)  # 0, 50, 100, ..., 1800
    
    cost = 0
    for state in states:
        lower = threshold - state
        upper = truck_1_capacity - state

        if lower <= state and upper >= state:
            truck_cost = truck_1_cost
        elif lower < state and upper < state:
            truck_cost = truck_2_cost
        else:
            truck_cost = 0
        cost += pi[int(state / 50)] * truck_cost
    # print(f"\nTotal Expected Truck Cost per Day: {cost:.6f}")
    return cost

def variable_3pl_shipping_cost(steady_state_csv, variable_shipment_cost, threshold):
    # Read the steady-state probabilities from CSV
    df = pd.read_csv(steady_state_csv, index_col=0)  # First column is state index (0, 50, ..., 1800)
    pi = df["Probability"].values  # Extract probability column
    # Verify length
    if len(pi) != 37:
        raise ValueError(f"Expected 37 states, got {len(pi)}")
    # Define states (inventory levels)
    states = np.arange(0, 1850, 50)  # 0, 50, 100, ..., 1800
    cost = 0
    # Calculate variable shipping cost: pi * state * variable_shipment_cost
    for state in states:
        if state >= threshold:
            cost += variable_shipment_cost * state * pi[int(state / 50)]
    # print(f"\nTotal Expected Variable Shipping Cost per Day: {cost:.6f}")
    return cost

def compute_total_cost_truck_method(threshold):
    # Example truck costs
    truck_1_cost = 1000  # Cost for truck 1
    truck_2_cost = 1500  # Cost for truck 2
    holding_cost_per_unit = 1.5/35.315  # Example holding cost per unit

    # Comppute the holding cost
    holding_cost = calculate_holding_cost(f"Threshold_Matrix_Steady_States/steady_state_{threshold}.csv", threshold=threshold, holding_cost_per_unit=holding_cost_per_unit)
    # print(f"\nTotal Expected Holding Cost per Day Trucks: {holding_cost:.6f}")
    # Compute the truck cost
    truck_cost = compute_truck_cost(f"Threshold_Matrix_Steady_States/steady_state_{threshold}.csv", threshold, truck_1_cost, truck_2_cost)
    
    total_cost = truck_cost + holding_cost
    print(f"\nTotal Expected Cost per Day Trucks: {total_cost:.6f}")
    return total_cost

def compute_total_cost_3pl_method(threshold):
    fixed_shipment_cost = 800
    variable_shipment_cost = 15/35.315  # Example variable shipment cost in $/ft³
    holding_cost_per_unit = 1.5/35.315  # Example holding cost per unit

    # Compute the holding cost
    holding_cost = calculate_holding_cost(f"Threshold_Matrix_Steady_States/steady_state_{threshold}.csv", threshold, holding_cost_per_unit)
    # print(f"\nTotal Expected Holding Cost per Day 3PL: {holding_cost:.6f}")

    # Compute the truck cost
    variable_cost = variable_3pl_shipping_cost(f"Threshold_Matrix_Steady_States/steady_state_{threshold}.csv", variable_shipment_cost, threshold=threshold)
    
    total_cost = holding_cost + variable_cost + fixed_shipment_cost


    print(f"\nTotal Expected Cost per Day 3PL: {total_cost:.6f}")
    return total_cost

compute_total_cost_truck_method(900) # Example threshold
compute_total_cost_3pl_method(900) # Example threshold

# Example usage
# steady_state_csv = "Threshold_Matrix_Steady_States/steady_state_800.csv"  # Replace with your actual file
# holding_cost_per_unit = 0.5083168  # Example: $0.0283168 per ft³ per day (from earlier context)
# total_cost = calculate_holding_cost(steady_state_csv, holding_cost_per_unit)
# truck_cost = compute_truck_cost(steady_state_csv, 800, 1000, 2000)  # Example truck costs
# Example usage
# Assuming your matrix is in "transition_matrix_200.csv"
# csv_file = "Threshold_Matrices/transition_matrix_300.csv"
# steady_state = compute_steady_state_from_csv(csv_file, "steady_state_200.csv")

