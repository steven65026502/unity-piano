import torch
import numpy as np
from data_processing import load_test_data
from model import CustomGPT2LMHeadModel, CustomGPT2Config
from torch.utils.data import DataLoader, TensorDataset

# Set device to GPU if available, otherwise to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the test data and labels
test_data_path = "C:/Users/z7913/Documents/project/unity-piano/test_pianoroll"
test_data, test_labels = load_test_data(test_data_path)

# Initialize the model and move it to the device
config = CustomGPT2Config()
model = CustomGPT2LMHeadModel(config).to(device)

# Load the trained model parameters
model.load_state_dict(torch.load("model.pth"))

# Convert the test data to a tensor and move it to the device
test_data_tensor = torch.tensor(test_data, dtype=torch.long).to(device)

# Use the model to make predictions
batch_size = 2
test_dataset = TensorDataset(test_data_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for step, inputs in enumerate(test_dataloader):
        inputs = inputs[0].to(device)
        outputs = model(input_ids=inputs[:, :-1], labels=inputs[:, 1:])
        logits = outputs[1]  # Extract the logits from the output tuple
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        labels = inputs[:, 1:].cpu().numpy()
        for i in range(len(preds)):
            if (labels[i] != 0).any():  # Filter out the padding part
                total += 1
                if np.array_equal(preds[i][labels[i] != 0], labels[i][labels[i] != 0]):
                    correct += 1

accuracy = correct / total
print("Accuracy: {:.2f}%".format(accuracy * 100))
