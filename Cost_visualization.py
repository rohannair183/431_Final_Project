import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import get_cost_for_matrix as gcm

# Compute costs and plot
thresholds = np.arange(300, 1850, 50)  # 50 to 1800 in steps of 50
truck_costs = []
pl_costs = []

for threshold in thresholds:
    truck_costs.append(gcm.compute_total_cost_truck_method(threshold))
    pl_costs.append(gcm.compute_total_cost_3pl_method(threshold))

# Create DataFrame for plotting
data = pd.DataFrame({
    "Threshold": np.concatenate([thresholds, thresholds]),
    "Cost": np.concatenate([truck_costs, pl_costs]),
    "Method": ["Truck"] * len(thresholds) + ["3PL"] * len(thresholds)
})

# Plot using Seaborn
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x="Threshold", y="Cost", hue="Method", marker="o")

# Customize plot
plt.title("Total Expected Cost per Day: Truck vs 3PL Methods")
plt.xlabel("Threshold (ft³)")
plt.ylabel("Cost ($ per day)")
plt.legend(title="Method")
plt.tight_layout()

# Save and show
plt.savefig("cost_comparison.png")
plt.show()