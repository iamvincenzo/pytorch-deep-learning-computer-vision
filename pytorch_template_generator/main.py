from pathlib import Path

CONFIG_FILE_PATH = "config.txt"

PROJECT_PATH = Path("project_template")
PROJECT_PATH.mkdir(parents=True, exist_ok=True)

PYTHON_FILENAME = "generated_template.py"
TEMPLATE_SAVE_PATH = PROJECT_PATH / PYTHON_FILENAME

# Read the config file
with open(CONFIG_FILE_PATH, "r") as config_file:
    config_content = config_file.read().strip()

# Define template components
common_imports = """import torch
import torch.nn as nn
import torch.optim as optim
"""

binary_classification_template = """
class BinaryClassificationModel(nn.Module):
    def __init__(self):
        super(BinaryClassificationModel, self).__init__()
        # Define your binary classification model architecture here

    def forward(self, x):
        # Define the forward pass
        return x
"""

multi_class_classification_template = """
class MultiClassClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(MultiClassClassificationModel, self).__init__()
        # Define your multi-class classification model architecture here
        self.fc = nn.Linear(in_features=your_input_size, out_features=num_classes)

    def forward(self, x):
        # Define the forward pass
        return x
"""

training_template = """
# Define your training loop here
def train(model, train_loader, criterion, optimizer, num_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print the average loss for this epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

# Example usage:
# model = BinaryClassificationModel()  # or MultiClassClassificationModel(num_classes=your_num_classes)
# train_loader = torch.utils.data.DataLoader(...)  # Your training data loader
"""

# Map config values to corresponding templates
template_mapping = {
    "binary": common_imports + binary_classification_template + training_template + """
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train(model, train_loader, criterion, optimizer)
""",
    "multiclass": common_imports + multi_class_classification_template + training_template + """
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train(model, train_loader, criterion, optimizer)
""",
    # Add more mappings as needed
}

# Check if the config value is in the mapping
if config_content in template_mapping:
    template_code = template_mapping[config_content]
else:
    template_code = "# Unsupported classification type"

# Write the Python code to a file
with open(TEMPLATE_SAVE_PATH, "w") as python_file:
    python_file.write(template_code)
