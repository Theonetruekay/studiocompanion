from torch.utils.data import random_split, DataLoader

# Determine dataset size
n_samples = tensor_mfccs.shape[0]

# Define ratios
train_size = int(0.7 * n_samples)
val_size = int(0.15 * n_samples)
test_size = n_samples - train_size - val_size

# Split the dataset
train_set, val_set, test_set = random_split(tensor_mfccs, [train_size, val_size, test_size])

# Create DataLoader instances
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)
