import numpy as np
import pandas as pd
import itertools
import math
from collections import defaultdict

# --- Parameters ---
component_volumes = {"A": 50, "B": 100}  # Volumes of components
component_probs = {"A": 0.4, "B":0.6}
lambda_orders = 2
max_daily_production = 3
volume_step = 50
max_volume = 900
due_days_probs = {2: 0.45, 3:0.55}  # Due days probabilities
max_due_days = max(due_days_probs.keys())  # = 3

# --- Poisson distribution for order arrivals ---
def generate_poisson_distribution(lambda_orders, max_orders=5):
    poisson_probs = [((lambda_orders ** k) * np.exp(-lambda_orders)) / math.factorial(k) for k in range(max_orders)]
    poisson_probs.append(1 - sum(poisson_probs))  # lump tail into last bin
    return poisson_probs

# --- Step 1: Joint (volume, due) distribution of daily arrivals (no aging yet) ---
joint_dist = defaultdict(float)
component_list = list(component_volumes.keys())
poisson_probs = generate_poisson_distribution(lambda_orders)

for n_orders, p_orders in enumerate(poisson_probs):
    actual_orders = min(n_orders, max_daily_production)
    if actual_orders == 0:
        joint_dist[(0, 0)] += p_orders
        continue

    comp_combos = list(itertools.product(component_list, repeat=actual_orders))
    due_combos = list(itertools.product(due_days_probs.keys(), repeat=actual_orders))

    for comp_combo in comp_combos:
        vol = sum(component_volumes[c] for c in comp_combo)
        p_comp = np.prod([component_probs[c] for c in comp_combo])

        for due_combo in due_combos:
            p_due = np.prod([due_days_probs[d] for d in due_combo])
            min_due = min(due_combo)  # No aging yet
            total_prob = p_orders * p_comp * p_due
            joint_dist[(vol, min_due)] += total_prob

# --- Step 2: Define state space (volume, due) ---
volume_states = list(range(0, max_volume + volume_step, volume_step))
due_states = list(range(max_due_days + 1))  # Includes due = 0, 1, 2, 3
state_space = [(v, d) for v in volume_states for d in due_states if not (v == 0 and d != 0)]
state_index = {state: i for i, state in enumerate(state_space)}
n_states = len(state_space)

# --- Step 3: Build transition matrix ---
P = np.zeros((n_states, n_states))

for (v, due), i in state_index.items():
    if due == 0 and v > 0:
        # Shipment occurs: reset to (0, 0)
        j = state_index[(0, 0)]
        P[i, j] = 1.0
        continue

    for (added_v, new_due), prob in joint_dist.items():
        new_total = v + added_v
        v_next = new_total if new_total <= max_volume else new_total - max_volume

        # Aging logic
        if v > 0:
            aged_due = max(due - 1, 0)
        else:
            aged_due = 0

        if added_v > 0:
            d_next = min(aged_due, new_due) if aged_due > 0 else new_due
        else:
            d_next = aged_due

        # Clamp and round
        v_next = min((v_next // volume_step) * volume_step, max_volume)
        d_next = max(min(d_next, max_due_days), 0)

        if v_next == 0:
            d_next = 0  # no inventory, so no due date

        next_state = (v_next, d_next)
        if next_state in state_index:
            j = state_index[next_state]
            P[i, j] += prob

# --- Step 4: Normalize transition matrix safely ---
row_sums = P.sum(axis=1, keepdims=True)
valid_rows = row_sums.flatten() > 0
P[valid_rows, :] = P[valid_rows, :] / row_sums[valid_rows]

# --- Step 5: Export to CSV ---
df = pd.DataFrame(P, index=[f"({v}, {d})" for v, d in state_space],
                  columns=[f"({v}, {d})" for v, d in state_space])
csv_path_updated = "final_transition_matrix_correct.csv"
df.to_csv(csv_path_updated, index_label="Current State (Volume, Due Date)")

