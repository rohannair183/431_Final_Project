import numpy as np
from scipy.stats import poisson
from collections import defaultdict
import itertools
import csv
import os # Import os module for path handling

print("Setting up parameters for states as multiples of 50...")

# --- Parameters ---
MAX_WEIGHT = 1800
STATE_INCREMENT = 50       # States are multiples of this value
# Calculate the number of states: 0, 50, 100, ..., 1800
NUM_STATES = MAX_WEIGHT // STATE_INCREMENT + 1
# Create a list of the actual state weight values
STATE_VALUES = [i * STATE_INCREMENT for i in range(NUM_STATES)]

LAMBDA = 2                 # Poisson parameter for orders per day
MAX_ORDERS_ACCEPTED = 3    # Maximum orders accepted (0, 1, 2, or 3)
COMPONENT_PROBS = {
    50: 0.3,   # Component A: weight 50, probability 0.3
    100: 0.5,  # Component B: weight 100, probability 0.5
    150: 0.2   # Component C: weight 150, probability 0.2
}
OUTPUT_CSV_FILE = 'markov_chain_transition_matrix_multiples_of_50.csv'

print(f"Max weight: {MAX_WEIGHT}")
print(f"State increment: {STATE_INCREMENT}")
print(f"Actual states: {STATE_VALUES}")
print(f"Number of states: {NUM_STATES}")
print(f"Poisson Lambda: {LAMBDA}, Max orders accepted: {MAX_ORDERS_ACCEPTED}")
print(f"Component probabilities: {COMPONENT_PROBS}")

# --- Helper function to map weight to state index ---
def weight_to_index(weight):
    # Ensure weight is a multiple of increment for safety, though it should be
    # if weight % STATE_INCREMENT != 0:
    #    print(f"Warning: Weight {weight} is not a multiple of {STATE_INCREMENT}")
    return weight // STATE_INCREMENT

# --- 1. Calculate Poisson Probabilities (Same as before) ---
print("Calculating Poisson probabilities...")
poisson_probs = {}
poisson_probs[0] = poisson.pmf(0, LAMBDA)
poisson_probs[1] = poisson.pmf(1, LAMBDA)
poisson_probs[2] = poisson.pmf(2, LAMBDA)
poisson_probs[3] = 1.0 - (poisson_probs[0] + poisson_probs[1] + poisson_probs[2])
# Ensure sum is close to 1
if not np.isclose(sum(poisson_probs.values()), 1.0):
     print(f"Warning: Poisson probabilities do not sum exactly to 1: {sum(poisson_probs.values())}. Adjusting P(>=3).")
     poisson_probs[3] = 1.0 - (poisson_probs[0] + poisson_probs[1] + poisson_probs[2])

print(f"Prob(0 orders): {poisson_probs[0]:.4f}")
print(f"Prob(1 order): {poisson_probs[1]:.4f}")
print(f"Prob(2 orders): {poisson_probs[2]:.4f}")
print(f"Prob(3+ orders): {poisson_probs[3]:.4f}")
print(f"Sum of Poisson probs: {sum(poisson_probs.values()):.4f}")


# --- 2. Calculate Probability Distribution of Added Weight (Same as before) ---
print("Calculating added weight distributions for 1, 2, 3 orders...")
# (This part doesn't need to change as component weights are already multiples of 50)
added_weight_dist_1 = defaultdict(float)
for weight, prob in COMPONENT_PROBS.items():
    added_weight_dist_1[weight] += prob

added_weight_dist_2 = defaultdict(float)
for order1, order2 in itertools.product(COMPONENT_PROBS.items(), repeat=2):
    weight1, prob1 = order1
    weight2, prob2 = order2
    total_weight = weight1 + weight2
    combined_prob = prob1 * prob2
    added_weight_dist_2[total_weight] += combined_prob

added_weight_dist_3 = defaultdict(float)
for order1, order2, order3 in itertools.product(COMPONENT_PROBS.items(), repeat=3):
    weight1, prob1 = order1
    weight2, prob2 = order2
    weight3, prob3 = order3
    total_weight = weight1 + weight2 + weight3
    combined_prob = prob1 * prob2 * prob3
    added_weight_dist_3[total_weight] += combined_prob

daily_added_weight_dists = {
    1: added_weight_dist_1,
    2: added_weight_dist_2,
    3: added_weight_dist_3,
}

# --- 3. Initialize and Populate the Transition Matrix ---
print(f"Initializing transition matrix ({NUM_STATES}x{NUM_STATES})...")
# Initialize matrix with zeros. Rows represent 'from state', columns 'to state'.
transition_matrix = np.zeros((NUM_STATES, NUM_STATES), dtype=float)

print("Populating transition matrix (this should be faster now)...")
# Iterate through each state *index*
for current_index in range(NUM_STATES):
    current_state_weight = STATE_VALUES[current_index] # Get the actual weight for this index

    # Case 0: Zero orders arrive (Probability P(0))
    prob_0_orders = poisson_probs[0]
    # The state remains the same, so the index remains the same
    next_index_0 = current_index
    transition_matrix[current_index, next_index_0] += prob_0_orders

    # Cases 1, 2, 3: Orders arrive (Probabilities P(1), P(2), P(3+))
    for num_orders in range(1, MAX_ORDERS_ACCEPTED + 1):
        prob_k_orders = poisson_probs[num_orders]
        added_weight_dist = daily_added_weight_dists[num_orders]

        # Iterate through the possible total weights added for this number of orders
        for added_weight, prob_added_weight in added_weight_dist.items():
            # Calculate the next state's actual weight, capped at MAX_WEIGHT
            next_state_weight = min(current_state_weight + added_weight, MAX_WEIGHT)

            # Convert the next state weight to its corresponding index
            next_index = weight_to_index(next_state_weight)

            # The probability of this specific transition is:
            # P(k orders) * P(specific total added weight | k orders)
            transition_prob = prob_k_orders * prob_added_weight
            transition_matrix[current_index, next_index] += transition_prob

print("Matrix population complete.")

# --- 4. Verification (Check Row Sums) ---
print("Verifying matrix row sums...")
row_sums = np.sum(transition_matrix, axis=1)
if np.allclose(row_sums, 1.0):
    print("Verification successful: All row sums are close to 1.0.")
else:
    print("Verification failed: Some row sums are not close to 1.0.")
    problematic_indices = np.where(~np.isclose(row_sums, 1.0))[0]
    problematic_states = [STATE_VALUES[i] for i in problematic_indices]
    print(f"Problematic states (weights): {problematic_states}")
    print(f"Sums for problematic states: {row_sums[problematic_indices]}")

# --- 5. Save the Matrix to CSV ---
output_path = os.path.join(os.getcwd(), OUTPUT_CSV_FILE) # Save in current directory
print(f"Saving transition matrix to '{output_path}'...")

try:
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header row using actual state weights
        header = [f'To_{state_val}' for state_val in STATE_VALUES]
        writer.writerow(['From_State'] + header) # Add a column label for the rows

        # Write matrix rows, labeling rows with actual state weights
        for i, row in enumerate(transition_matrix):
            from_state_label = f'From_{STATE_VALUES[i]}'
            writer.writerow([from_state_label] + list(row))

    print(f"Successfully saved matrix to {output_path}")

except IOError as e:
    print(f"Error saving file: {e}")
except Exception as e:
    print(f"An unexpected error occurred during saving: {e}")

print("Script finished.")