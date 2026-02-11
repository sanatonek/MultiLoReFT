import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Use seaborn style
sns.set(style="whitegrid", font_scale=1.2)

# Labels
groups = [r'1', r'2']  # modalities
bar_labels = [r'$h$', r'$\phi$', r'$c$']  # methods

# Updated numbers from new table
metric1_means = np.array([
    [0.101, 0.069, 0.096],  # group 1: h1, phi1, c1
    [0.060, 0.031, 0.059]   # group 2: h2, phi2, c2
])
metric1_stds = np.array([
    [0.008, 0.006, 0.010],
    [0.007, 0.005, 0.008]
])

metric2_means = np.array([
    [0.483, 0.488, 0.447],  # group 1
    [0.272, 0.289, 0.276]   # group 2
])
metric2_stds = np.array([
    [0.039, 0.029, 0.021],
    [0.030, 0.022, 0.026]
])

x = np.arange(len(groups))  # group positions
width = 0.2

fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey=False)

colors = ['#4c72b0', '#55a868', '#c44e52']  # nice seaborn palette

# ---- Metric 1 subplot ----
for i in range(len(bar_labels)):
    axes[0].bar(x + (i-1)*width, metric1_means[:, i], width,
                yerr=metric1_stds[:, i], capsize=5,
                label=bar_labels[i], color=colors[i])
axes[0].set_xticks(x)
axes[0].set_xticklabels([f'Modality {g}' for g in groups], rotation=30, fontsize=14)
axes[0].set_title("Joint label (Simulation I)", fontsize=18)
axes[0].set_ylabel("Score", fontsize=14)
# axes[0].set_xlabel("Modality", fontsize=14)
axes[0].legend(fontsize=14)

# ---- Metric 2 subplot ----
for i in range(len(bar_labels)):
    axes[1].bar(x + (i-1)*width, metric2_means[:, i], width,
                yerr=metric2_stds[:, i], capsize=5,
                label=bar_labels[i], color=colors[i])
axes[1].set_xticks(x)
axes[1].set_xticklabels([f'Modality {g}' for g in groups], rotation=30, fontsize=14)
axes[1].set_title("Emotion label (Crema-D)", fontsize=18)
axes[1].set_ylabel("Score", fontsize=14)
# axes[1].set_xlabel("Modality", fontsize=14, rotation=30)
axes[1].legend(fontsize=14)

# plt.suptitle("Fine-tuning Improves Cross-modal Predictions", fontsize=14)
plt.tight_layout()
plt.savefig('./plots/cross-modal-predictions.pdf')
