import create_threshold_matrix as ct
import get_cost_for_matrix as gcm


import numpy as np
import pandas as pd
import os
import itertools


for threshold in range(300, 1801, 50):
    print(f"Generating transition matrix for threshold: {threshold}")

    # Check if the output directory exists, if not create it
    output_dir = "Threshold_Matrices"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    

    output_file = f"{output_dir}/transition_matrix_{threshold}.csv"
    # Create and save transition matrix for the given threshold
    transition_matrix = ct.create_transition_matrix(threshold, output_file=output_file)
    
    # Compute steady-state probabilities and save to CSV
    steady_state = gcm.compute_steady_state_from_csv(output_file, f"Threshold_Matrix_Steady_States/steady_state_{threshold}.csv")
    
    print(f"Transition matrix and steady state for threshold {threshold} saved.")