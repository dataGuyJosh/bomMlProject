import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the CSV data
data = pd.read_csv('data/raw_data.csv')

# Replace non-numeric values with zeros for relevant columns
non_numeric_cols = ['9am wind speed (km/h)']
for col in non_numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    data[col].fillna(0, inplace=True)

# drop other null values
data.dropna(inplace=True)

# Convert '9am wind direction' to one-hot encoding
data = pd.get_dummies(data, columns=['9am wind direction'])

# Extract relevant columns
features = ['9am Temperature (°C)', '9am relative humidity (%)', '9am cloud amount (oktas)',
            '9am wind speed (km/h)', '9am MSL pressure (hPa)'] + [col for col in data.columns if '9am wind direction_' in col]
target = '3pm Temperature (°C)'

# Extract features and target
X = data[features].values
y = data[target].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# Define a simple feedforward neural network
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create the model and set the loss function and optimizer
model = Net(input_size=len(features))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train.view(-1, 1))
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    mse = nn.MSELoss()
    test_loss = mse(test_outputs, y_test.view(-1, 1))
    print(f'Test MSE: {test_loss.item():.4f}')
