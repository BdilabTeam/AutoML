import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sa_dataloader
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
dim_state = 136
dim_action = 24
hidden_size = 32
num_classes = 1
num_epochs = 10
batch_size = 1
learning_rate = 0.001

# MNIST dataset
train_dataset = sa_dataloader.SaDataset('offline-sa-train_pkl')
test_dataset = sa_dataloader.SaDataset('offline-sa-test_pkl')

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False)


# Fully connected neural network with one hidden layer
class Policy(nn.Module):
    def __init__(self, dim_state, hidden_size, dim_action):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(dim_state, hidden_size)

        self.action_head = nn.Linear(hidden_size, dim_action)
        self.value_head = nn.Linear(hidden_size, 1) # Scalar Value

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_score = self.action_head(x)
        state_value = self.value_head(x)

        return F.softmax(action_score, dim=-1), state_value


model = Policy(dim_state, hidden_size, dim_action+1).to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (x, y) in enumerate(train_loader):
        # Move tensors to the configured device
        x = x.float().to(device)
        y = y.float().to(device)
        # Forward pass
        action, state_value = model(x)
        loss = criterion(action, y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)\
total = 0
correct = 0
nonzero_total = 0
nonzero_correct = 0

with torch.no_grad():
    loss_list = []
    for x, y in test_loader:
        x = x.float().to(device)
        y = y.float().to(device)
        outputs, state_value = model(x)
        loss = criterion(outputs, y)
        loss_list.append(loss)
        total += 1
        correct += int((torch.max(y, 1)[1] == torch.max(outputs.data, 1)[1]))
        if torch.max(y, 1)[1] != 0:
            print(torch.max(y, 1)[1])
            nonzero_total += 1
            nonzero_correct += int((torch.max(y, 1)[1] == torch.max(outputs.data, 1)[1]))

    print('accuracy of the network: {}'.format(correct/total))
    print('non zero accuracy of the network: {}'.format(nonzero_correct / nonzero_total))


# Save the model checkpoint
torch.save(model.state_dict(), 'model32.ckpt')