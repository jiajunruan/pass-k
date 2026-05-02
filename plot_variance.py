import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

# 1. Load the data
# Replace with your actual CSV filename if it is different
csv_filename = 'all_seeds_pass_at_k.csv'
df = pd.read_csv(csv_filename)

# Ensure we are only looking at the 'k' columns, ignoring 'Seed'
k_columns = ['1', '2', '4', '8', '16', '32', '64', '128']
data = df[k_columns]

# 2. Calculate Statistical Metrics
means = data.mean()
stds = data.std()
n = len(data)

# Calculate 95% Confidence Interval using t-distribution
# CI = t-score * (standard_deviation / sqrt(n))
t_score = st.t.ppf(0.975, n - 1) 
cis = t_score * (stds / np.sqrt(n))

# 3. Create a Formatted Summary Table
summary_df = pd.DataFrame({
    'k': k_columns,
    'Mean': means,
    'Std Dev': stds,
    '95% CI': cis
})

# Create the presentation strings (Mean ± Metric)
summary_df['Mean ± Std'] = summary_df.apply(lambda row: f"{row['Mean']:.4f} ± {row['Std Dev']:.4f}", axis=1)
summary_df['Mean ± 95% CI'] = summary_df.apply(lambda row: f"{row['Mean']:.4f} ± {row['95% CI']:.4f}", axis=1)

print("-" * 50)
print("STATISTICAL SUMMARY TABLE")
print("-" * 50)
print(summary_df[['k', 'Mean ± Std', 'Mean ± 95% CI']].to_string(index=False))
print("-" * 50)

# 4. Generate the Figure
plt.figure(figsize=(10, 6))

# Convert k column names to integers for the x-axis
x_values = [int(k) for k in k_columns]
y_means = means.values
y_stds = stds.values

# Plot the mean line
plt.plot(x_values, y_means, marker='o', color='#1f77b4', linewidth=2, label='Mean Leak@k')

# Create a shaded region for the Standard Deviation
# (You can swap y_stds with cis.values if you prefer plotting the 95% CI instead)
plt.fill_between(x_values, y_means - y_stds, y_means + y_stds, color='#1f77b4', alpha=0.2, label='± 1 Std Dev')

# Formatting the plot
plt.xscale('log', base=2) # Base 2 log scale is perfect for 1, 2, 4, 8...
plt.xticks(x_values, k_columns) # Force x-ticks to exactly match your k values

plt.xlabel('k', fontsize=12, fontweight='bold')
plt.ylabel('Leak@k Score', fontsize=12, fontweight='bold')
plt.title('Leak@k Performance Across 24 Seeds', fontsize=14, fontweight='bold')

# Set Y-axis from 0 to 1 to show absolute scale, which makes the tiny variance look even smaller!
plt.ylim(0, 1.0) 

plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend(loc='lower right', fontsize=11)
plt.tight_layout()

# Save and display the figure
output_image = 'leak_at_k_variance.png'
plt.savefig(output_image, dpi=300)
print(f"✅ Figure successfully saved as {output_image}")

plt.show()