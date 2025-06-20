# %%
import matplotlib.pyplot as plt

# Define the metrics
metrics = {
    "accuracy": 0.7584,
    "precision": 0.6926,
    "recall": 0.8915,
    "F1 score": 0.7796
}

colors = ['skyblue', 'lightgreen', 'plum', 'salmon']

# Plot
plt.figure(figsize=(8, 5))
bars = plt.bar(metrics.keys(), metrics.values(), color=colors)

# Title and axis
plt.title("Test Metrics (Excluding Loss)", fontsize=16, weight='bold')
plt.ylabel("Score", fontsize=13, weight='bold')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Axis ticks
plt.xticks(fontsize=12, weight='bold')
plt.yticks(fontsize=12, weight='bold')
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)

# Annotate values
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{height:.3f}",
             ha='center', va='bottom', fontsize=11, weight='bold')

plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt

# F1 scores to compare
f1_scores = {
    "UNI": 0.8310,
    "CTranspath": 0.9095
}

colors = ['steelblue', 'darkorange']

# Plot
plt.figure(figsize=(6, 6))
bars = plt.bar(f1_scores.keys(), f1_scores.values(), color=colors)

# Title and axis
plt.title("F1 Score Comparison", fontsize=16, weight='bold')
plt.ylabel("F1 Score", fontsize=13, weight='bold')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Axis ticks
plt.xticks(fontsize=12, weight='bold')
plt.yticks(fontsize=12, weight='bold')
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)

# Annotate values
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{height:.4f}",
             ha='center', va='bottom', fontsize=11, weight='bold')

plt.tight_layout()
plt.show()
