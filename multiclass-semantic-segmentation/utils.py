import torch

# Function to calculate class weights
def calculate_class_weights(dataloader, num_classes):
    class_counts = torch.zeros(num_classes)

    # Calculate class frequencies
    for _, masks in dataloader:
        for class_idx in range(num_classes):
            class_counts[class_idx] += torch.sum(masks == class_idx).item()

    # Calculate inverse class frequencies
    inverse_class_frequencies = torch.where(class_counts > 0, 1 / class_counts, 0)

    # Normalize weights
    weights = inverse_class_frequencies / inverse_class_frequencies.sum()

    return weights