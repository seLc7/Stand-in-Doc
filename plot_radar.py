import matplotlib.pyplot as plt
import numpy as np

# Define dimensions in English
labels = [
    "Knowledge Depth",
    "Structure Analysis",
    "Semantic Understanding",
    "Style Detection",
    "Scenario Adaptation",
    "Context Awareness",
]

num_vars = len(labels)

# Scores for each attacker level (0-5)
scores = {
    "Level 0": [0, 1, 0, 0, 0, 1],
    "Level 1": [2, 3, 1, 1, 1, 2],
    "Level 2": [4, 4, 4, 3, 3, 4],
    "Level 3": [5, 5, 5, 5, 5, 5],
}

# Calculate angles for radar chart
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Close the circle

# Create radar chart
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

# Define new color scheme for better visibility
colors = ["#1F78B4", "forestgreen", "darkorange", "crimson"]

for idx, (level, values) in enumerate(scores.items()):
    values += values[:1]
    ax.plot(angles, values, label=level, color=colors[idx], linewidth=2, zorder=4)
    ax.fill(angles, values, color=colors[idx], alpha=0.15)

# Adjust labels and settings
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=14, zorder=6)

ax.set_ylim(0, 5)

# Add legend and title
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
# plt.title("Attacker Capability Radar Chart (Level 0-3)", size=16)

plt.tight_layout()
plt.show()
