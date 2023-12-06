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
dim_state = 209
dim_action = 3
hidden_size = 32
num_classes = 1
num_epochs = 50
batch_size = 4
learning_rate = 0.0001

# MNIST dataset
train_dataset = sa_dataloader.SaDataset('offline-sche-train_pkl')
test_dataset = sa_dataloader.SaDataset('offline-sche-test_pkl')

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True, drop_last=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False, drop_last=True)


# Fully connected neural network with one hidden layer
class Policy(nn.Module):
    def __init__(self, dim_state, hidden_size, dim_action):
        super(Policy, self).__init__()
        # self.ln = nn.LayerNorm(dim_state)
        self.bn1 = nn.BatchNorm1d(dim_state)
        self.fc1 = nn.Linear(dim_state, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.action_head = nn.Linear(hidden_size, dim_action)


    def forward(self, x):
        x = self.bn1(x)
        x = self.bn2(torch.tanh(self.fc1(x)))
        x = self.bn3(torch.tanh(self.fc2(x)))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


model = Policy(dim_state, hidden_size, dim_action).to(device)
print(model.state_dict())
# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

# Train the model
total_step = len(train_loader)
model.train()
for epoch in range(num_epochs):
    for i, (x, y) in enumerate(train_loader):
        # Move tensors to the configured device
        x = x.float().to(device)
        y = y.float().to(device)
        # Forward pass

        action = model(x)
        # print(action)
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

model.eval()
with torch.no_grad():
    loss_list = []
    for x, y in test_loader:
        x = x.float().to(device)
        y = y.float().to(device)
        outputs = model(x)
        print(outputs)
        loss = criterion(outputs, y)
        loss_list.append(loss)
        total += 1
        correct += int((torch.max(y, 1)[1] == torch.max(outputs.data, 1)[1]))
        if torch.max(y, 1)[1] != 0:
            # print(torch.max(y, 1)[1])
            nonzero_total += 1
            nonzero_correct += int((torch.max(y, 1)[1] == torch.max(outputs.data, 1)[1]))

    print('accuracy of the network: {}'.format(correct/total))
    print('non zero accuracy of the network: {}'.format(nonzero_correct / nonzero_total))


# Save the model checkpoint
torch.save(model.state_dict(), 'model32.ckpt')