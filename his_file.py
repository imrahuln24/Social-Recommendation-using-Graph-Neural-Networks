import pickle
import matplotlib.pyplot as plt

# Load the .his file
with open("History/amazon.his", "rb") as file:
    data = pickle.load(file)

# Extract data
train_loss = data["TrainLoss"]
train_pre_loss = data["TrainpreLoss"]
test_hr = data["TestHR"]
test_ndcg = data["TestNDCG"]

# Create x-axis (index)
epochs = range(len(train_loss))  # Assuming all lists have the same length

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 2 rows, 2 columns

# Plot Train Loss
axes[0, 0].plot(epochs, train_loss, marker="o", color="b")
axes[0, 0].set_title("Train Loss")
axes[0, 0].set_xlabel("Epochs")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].grid(True)

# Plot Train Pre-Loss
axes[0, 1].plot(epochs, train_pre_loss, marker="s", color="g")
axes[0, 1].set_title("Train Pre-Loss")
axes[0, 1].set_xlabel("Epochs")
axes[0, 1].set_ylabel("Loss")
axes[0, 1].grid(True)

# Plot Test HR
axes[1, 0].plot(epochs, test_hr, marker="^", color="r")
axes[1, 0].set_title("Test HR")
axes[1, 0].set_xlabel("Epochs")
axes[1, 0].set_ylabel("HR")
axes[1, 0].grid(True)

# Plot Test NDCG
axes[1, 1].plot(epochs, test_ndcg, marker="d", color="m")
axes[1, 1].set_title("Test NDCG")
axes[1, 1].set_xlabel("Epochs")
axes[1, 1].set_ylabel("NDCG")
axes[1, 1].grid(True)

# Adjust layout
plt.tight_layout()
plt.show()
