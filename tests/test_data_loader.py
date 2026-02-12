"""Test data loader with visualizations."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.data_loader import load_data, create_sequences, FEATURES

# Load data
print("=" * 60)
print("LOADING DATA")
print("=" * 60)
data = load_data("AAPL")

print(f"\nFeatures: {FEATURES}")
print(f"\nTrain shape: {data.train.shape}")
print(f"Val shape: {data.val.shape}")
print(f"Test shape: {data.test.shape}")

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 24))

# 1. Time series of all features (test set)
ax1 = fig.add_subplot(4, 2, 1)
data.test[FEATURES].plot(ax=ax1, alpha=0.7)
ax1.set_title("All Features Over Time (Test Set)")
ax1.set_xlabel("Date")
ax1.set_ylabel("% Change")
ax1.legend(loc='upper right', fontsize=8)
ax1.grid(True, alpha=0.3)

# 2. Target distribution
ax2 = fig.add_subplot(4, 2, 2)
data.train["Target"].hist(bins=50, ax=ax2, alpha=0.7, label="Train", color="blue")
data.test["Target"].hist(bins=50, ax=ax2, alpha=0.7, label="Test", color="orange")
ax2.set_title("Target Distribution (Next Day Return)")
ax2.set_xlabel("% Change")
ax2.set_ylabel("Frequency")
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Correlation heatmap
ax3 = fig.add_subplot(4, 2, 3)
corr = data.train[FEATURES + ["Target"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax3)
ax3.set_title("Feature Correlation Matrix (Train)")

# 4. Feature distributions
ax4 = fig.add_subplot(4, 2, 4)
data.train[FEATURES].boxplot(ax=ax4)
ax4.set_title("Feature Distributions (Train)")
ax4.set_ylabel("% Change")
ax4.tick_params(axis='x', rotation=45)
ax4.grid(True, alpha=0.3)

# 5. Close vs Target scatter
ax5 = fig.add_subplot(4, 2, 5)
ax5.scatter(data.train["Close"], data.train["Target"], alpha=0.3, s=5)
ax5.set_title("Close vs Target (Train)")
ax5.set_xlabel("Close (% change today)")
ax5.set_ylabel("Target (% change tomorrow)")
ax5.grid(True, alpha=0.3)

# 6. VXN vs Target scatter
ax6 = fig.add_subplot(4, 2, 6)
ax6.scatter(data.train["VXN"], data.train["Target"], alpha=0.3, s=5, color="red")
ax6.set_title("VXN vs Target (Train)")
ax6.set_xlabel("VXN (% change)")
ax6.set_ylabel("Target (% change tomorrow)")
ax6.grid(True, alpha=0.3)

# 7. Cumulative returns
ax7 = fig.add_subplot(4, 2, 7)
cumret_train = (1 + data.train["Target"]).cumprod()
cumret_test = (1 + data.test["Target"]).cumprod()
cumret_train.plot(ax=ax7, label="Train", color="blue")
cumret_test.plot(ax=ax7, label="Test", color="orange")
ax7.set_title("Cumulative Returns (Buy & Hold)")
ax7.set_xlabel("Date")
ax7.set_ylabel("Cumulative Return")
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. Sample data table
ax8 = fig.add_subplot(4, 2, 8)
ax8.axis('off')
sample = data.test.head(10).round(4)
table = ax8.table(
    cellText=sample.values,
    colLabels=sample.columns,
    rowLabels=sample.index.strftime('%Y-%m-%d'),
    loc='center',
    cellLoc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1.2, 1.5)
ax8.set_title("Sample Data (Test Set - First 10 Rows)", y=0.95)

plt.tight_layout()
plt.savefig("outputs/data_analysis.png", dpi=150, bbox_inches='tight')
print("\nSaved: outputs/data_analysis.png")

# ============================================================
# LSTM Sequences visualization
# ============================================================
print("\n" + "=" * 60)
print("LSTM SEQUENCES")
print("=" * 60)

X_train, y_train = create_sequences(data.train, seq_length=60)
X_test, y_test = create_sequences(data.test, seq_length=60)

print(f"X_train shape: {X_train.shape}  (samples, timesteps, features)")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Visualize one sequence
fig2, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

sample_idx = 100  # Pick a sample sequence
for i, feat in enumerate(FEATURES):
    ax = axes[i]
    ax.plot(X_train[sample_idx, :, i])
    ax.set_title(f"{feat} (60-day sequence)")
    ax.set_xlabel("Day")
    ax.set_ylabel("% Change")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

plt.suptitle(f"Sample LSTM Sequence (index={sample_idx}), Target={y_train[sample_idx]:.4f}", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/lstm_sequence.png", dpi=150, bbox_inches='tight')
print("Saved: outputs/lstm_sequence.png")

# ============================================================
# Feature importance preview (correlation with target)
# ============================================================
print("\n" + "=" * 60)
print("FEATURE CORRELATION WITH TARGET")
print("=" * 60)

corr_with_target = data.train[FEATURES].corrwith(data.train["Target"]).sort_values()
print(corr_with_target.to_string())

fig3, ax = plt.subplots(figsize=(10, 6))
corr_with_target.plot(kind='barh', ax=ax, color=['red' if x < 0 else 'green' for x in corr_with_target])
ax.set_title("Feature Correlation with Target")
ax.set_xlabel("Correlation")
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig("outputs/feature_correlation.png", dpi=150, bbox_inches='tight')
print("Saved: outputs/feature_correlation.png")

# ============================================================
# Statistics
# ============================================================
print("\n" + "=" * 60)
print("DATA STATISTICS (Train)")
print("=" * 60)
print(data.train.describe().round(4).to_string())

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
plt.show()
