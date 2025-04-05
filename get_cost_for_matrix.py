import numpy as np
import pandas as pd
from scipy.stats import poisson
import create_threshold_matrix as cvp
import itertools
from itertools import product
# Calculate volume probabilities for orders
volume_probabilities = {}
for i in range(1, 4):
    # Calculate volume probabilities for the current order
    volume_probabilities[i] = cvp.compute_volume_probabilities(i)
    print(f"Volume probabilities for {i} orders: {volume_probabilities[i]}")
def compute_steady_state_from_csv(csv_file, output_file="steady_state_probabilities.csv"):
    # Read the transition matrix from CSV
    df = pd.read_csv(csv_file, index_col=0)  # First column is index (states)
    P = df.values  # Convert to NumPy array
        
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
    steady_state = pd.Series(0)
    try:
        steady_state = pd.Series(pi, index=states, name="Probability")
    except ValueError:
        print("Warning: Length of pi does not match number of states.")
        states = df.columns
        state_text = [str(s) for s in states]

        print("States from CSV:", state_text)
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

def calculate_holding_cost(steady_state_csv, threshold, holding_cost_per_unit,late_policy=False):
    # Read the steady-state probabilities from CSV
    df = pd.read_csv(steady_state_csv, index_col=0)  # First column is state index (0, 50, ..., 1800)
    pi = df["Probability"].values  # Extract probability column
    
    # Define states (inventory levels)
    states = np.arange(0, 1850, 50)  # 0, 50, 100, ..., 1800
    if late_policy:
        # Define inventory levels
        inventory_levels = np.arange(0, 950, 50)  # 0, 50, 100, ..., 900 (18 levels)
        time_values = np.arange(0, 4)             # 0, 1, 2, 3 (4 time states) - used for count only

        # Create states with only inventory levels, repeated for each time value
        states = np.repeat(inventory_levels, len(time_values)).reshape(-1, 1)

        # Convert to tuples if needed (optional)
        states = [state[0] for state in states]
        states = states[3:]
    
    # Calculate holding cost: pi * state * holding_cost_per_unit
    holding_costs = pi * states * holding_cost_per_unit
    total_holding_cost_per_day = np.sum(holding_costs)
    p_orders = poisson.pmf([0, 1, 2, 3], 2)
    p_orders[3] = 1 - sum(p_orders[:3])  # Adjust P(N=3) for cap
    orders_per_day = (1 * p_orders[1] + 2 * p_orders[2] + 3 * p_orders[3])
    volume_per_order = 0.3 * 50 + 0.5 * 100 + 0.2 * 150
    cycle_length = threshold / (orders_per_day * volume_per_order)
    total_holding_cost = total_holding_cost_per_day * cycle_length

    # print(f"\nTotal Expected Holding Cost per Day: {total_holding_cost/cycle_length:.6f}")
    return total_holding_cost_per_day

def compute_truck_cost(steady_state_csv, threshold, truck_1_cost, truck_2_cost, truck_1_capacity=900, truck_2_capacity=1800, late_policy=False):
    # Read the steady-state probabilities from CSV
    df = pd.read_csv(steady_state_csv, index_col=0)  # First column is state index (0, 50, ..., 1800)
    
    pi = df["Probability"].values  # Extract probability column
    # Define states (inventory levels)
    states = np.arange(threshold-450, threshold-49, 50)
    possible_orders = np.arange(50, 451, 50)  # 0, 50, 100, ..., 450
    # print(f"States: {states}")
    # print(f"Orders: {possible_orders}")
    p_orders = poisson.pmf([0, 1, 2, 3], 2)
    p_orders[3] = 1 - sum(p_orders[:3])  # Adjust P(N=3) for cap

    cost = 0

    # Calculate truck cost: pi * state * truck_cost
    if late_policy:
        # Define inventory levels
        inventory_levels = np.arange(0, 950, 50)
        # 0, 50, 100, ..., 900 (18 levels)
        time_values = np.arange(0, 4)             # 0, 1, 2, 3 (4 time states) - used for count only
        # Create states with only inventory levels, repeated for each time value
        states = np.repeat(inventory_levels, len(time_values)).reshape(-1, 1)
        # Convert to tuples if needed (optional)
        states = [state[0] for state in states]
        states = states[3:]

    for state in states:
        for posible_order in possible_orders:
            if state + posible_order >= threshold:
                # Calculate the cost based on the truck capacity
                if state <= truck_1_capacity:
                    truck_cost = truck_1_cost
                elif state <= truck_2_capacity:
                    truck_cost = truck_2_cost
                else:
                    truck_cost = 0
                # Calculate the expected cost
                cost += (pi[int(state / 50)] * state + posible_order * (volume_probabilities[1].get(posible_order, 0) * p_orders[1] + volume_probabilities[2].get(posible_order, 0) * p_orders[2] + volume_probabilities[3].get(posible_order, 0) * p_orders[3])) * truck_cost
    
    
    orders_per_day = (1 * p_orders[1] + 2 * p_orders[2] + 3 * p_orders[3])
    volume_per_order = 0.3 * 50 + 0.5 * 100 + 0.2 * 150
    cycle_length = threshold / (orders_per_day * volume_per_order)
    cost_per_day = cost / cycle_length
    return cost_per_day

def variable_3pl_shipping_cost(steady_state_csv, variable_shipment_cost, threshold, late_policy=False):
    # Read the steady-state probabilities from CSV
    df = pd.read_csv(steady_state_csv, index_col=0)  # First column is state index (0, 50, ..., 1800)
    pi = df["Probability"].values  # Extract probability column
        
    # Define states (inventory levels)
    states = np.arange(threshold-450, threshold-49, 50)
    if late_policy:
        # Define inventory levels
        inventory_levels = np.arange(0, 950, 50)
        # 0, 50, 100, ..., 900 (18 levels)
        time_values = np.arange(0, 4)             # 0, 1, 2, 3 (4 time states) - used for count only
        # Create states with only inventory levels, repeated for each time value
        states = np.repeat(inventory_levels, len(time_values)).reshape(-1, 1)
        # Convert to tuples if needed (optional)
        states = [state[0] for state in states]
        states = states[3:]
    
    # print(f"States: {states}")
    orders = np.arange(50, 451, 50)  # 0, 50, 100, ..., 450
    # print(f"Orders: {orders}")
        
    cost = 0
    p_orders = poisson.pmf([0, 1, 2, 3], 2)
    p_orders[3] = 1 - sum(p_orders[:3])  # Adjust P(N=3) for cap
    # Calculate variable shipping cost: pi * state * variable_shipment_cost
    for i in range(len(states)):
        state = states[i]
        for j in range(len(orders)):
            order = orders[j]
            if state + order >= threshold:
                # print(f"probability of {order}: {volume_probabilities[1].get(order, 0) + volume_probabilities[2].get(order, 0) + volume_probabilities[3].get(order, 0)}")
                # print(pi[int(state // 50)], state, order)
                cost += ((pi[int(state // 50)]*state) + order * (volume_probabilities[1].get(order, 0) * p_orders[1] + volume_probabilities[2].get(order, 0) * p_orders[2] + volume_probabilities[3].get(order, 0) * p_orders[3])) * variable_shipment_cost
                # print(pi[int(state // 50)], state, order)
    # print(f"\nTotal Expected Variable Shipping Cost per Day: {cost:.6f}")
    
    orders_per_day = (1 * p_orders[1] + 2 * p_orders[2] + 3 * p_orders[3])
    volume_per_order = 0.3 * 50 + 0.5 * 100 + 0.2 * 150
    cycle_length = threshold / (orders_per_day * volume_per_order)
    # print(f"cost per cycle {cost / cycle_length}")
    return cost / cycle_length

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
    p_orders = poisson.pmf([0, 1, 2, 3], 2)
    p_orders[3] = 1 - sum(p_orders[:3])  # Adjust P(N=3) for cap
    orders_per_day = (1 * p_orders[1] + 2 * p_orders[2] + 3 * p_orders[3])
    volume_per_order = 0.3 * 50 + 0.5 * 100 + 0.2 * 150
    cycle_length = threshold / (orders_per_day * volume_per_order)
    fixed_shipment_cost = 800 / cycle_length  # Example fixed shipment cost in $/day
    variable_shipment_cost = 15 * 35.3147  # Example variable shipment cost in $/ft³
    holding_cost_per_unit = 1.5 * 35.3147  # Example holding cost per unit

    # Compute the holding cost
    holding_cost = calculate_holding_cost(f"Threshold_Matrix_Steady_States/steady_state_{threshold}.csv", threshold, holding_cost_per_unit)
    # print(f"\nTotal Expected Holding Cost per Day 3PL: {holding_cost:.6f}")

    # Compute the truck cost
    variable_cost = variable_3pl_shipping_cost(f"Threshold_Matrix_Steady_States/steady_state_{threshold}.csv", variable_shipment_cost, threshold)
    
    total_cost = holding_cost + variable_cost + fixed_shipment_cost


    print(f"\nTotal Expected Cost per Cycle 3PL: {total_cost:.6f}")
    return total_cost

def compute_total_truck_cost_no_delay():
    # Example truck costs
    truck_1_cost = 1000  # Cost for truck 1
    truck_2_cost = 1500  # Cost for truck 2
    holding_cost_per_unit = 1.5/35.315  # Example holding cost per unit

    compute_steady_state_from_csv("final_transition_matrix_correct.csv", "steady_state_no_late.csv")
    # Comppute the holding cost
    holding_cost = calculate_holding_cost("steady_state_no_late.csv", threshold=900, holding_cost_per_unit=holding_cost_per_unit, late_policy=True)
    # print(f"\nTotal Expected Holding Cost per Day Trucks: {holding_cost:.6f}")
    # Compute the truck cost
    truck_cost = compute_truck_cost("steady_state_no_late.csv", 900, truck_1_cost, truck_2_cost)
    
    total_cost = truck_cost + holding_cost
    print(f"\nTotal Expected Cost per Day Trucks No Delay: {total_cost:.6f}")
    return total_cost

def compute_total_cost_3pl_method_no_delay():
    p_orders = poisson.pmf([0, 1, 2, 3], 2)
    p_orders[3] = 1 - sum(p_orders[:3])  # Adjust P(N=3) for cap
    orders_per_day = (1 * p_orders[1] + 2 * p_orders[2] + 3 * p_orders[3])
    volume_per_order = 0.3 * 50 + 0.5 * 100 + 0.2 * 150
    cycle_length = 900 / (orders_per_day * volume_per_order)
    fixed_shipment_cost = 800 / cycle_length  # Example fixed shipment cost in $/day
    variable_shipment_cost = 15 * 35.3147  # Example variable shipment cost in $/ft³
    holding_cost_per_unit = 1.5 * 35.3147  # Example holding cost per unit

    # Compute the holding cost
    holding_cost = calculate_holding_cost("steady_state_no_late.csv", threshold=900, holding_cost_per_unit=holding_cost_per_unit, late_policy=True)
    # print(f"\nTotal Expected Holding Cost per Day Trucks: {holding_cost:.6f}")

    # Compute the truck cost
    variable_cost = variable_3pl_shipping_cost("steady_state_no_late.csv", variable_shipment_cost, threshold=900, late_policy=True)
    
    total_cost = holding_cost + variable_cost + fixed_shipment_cost


    print(f"\nTotal Expected Cost per Cycle 3PL No Delay: {total_cost:.6f}")
    return total_cost
# compute_total_cost_truck_method(900) # Example threshold
compute_total_truck_cost_no_delay()
compute_total_cost_3pl_method_no_delay()
# compute_total_cost_3pl_method(900) # Example threshold

# Example usage
# steady_state_csv = "Threshold_Matrix_Steady_States/steady_state_800.csv"  # Replace with your actual file
# holding_cost_per_unit = 0.5083168  # Example: $0.0283168 per ft³ per day (from earlier context)
# total_cost = calculate_holding_cost(steady_state_csv, holding_cost_per_unit)
# truck_cost = compute_truck_cost(steady_state_csv, 800, 1000, 2000)  # Example truck costs
# Example usage
# Assuming your matrix is in "transition_matrix_200.csv"
# csv_file = "Threshold_Matrices/transition_matrix_300.csv"
# steady_state = compute_steady_state_from_csv(csv_file, "steady_state_200.csv")

