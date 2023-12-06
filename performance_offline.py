import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sa_dataloader
import torch.nn.functional as F

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 1099
hidden_size = 128
num_classes = 1
num_epochs = 1
batch_size = 8
learning_rate = 0.001

# MNIST dataset
train_dataset = sa_dataloader.SaDataset('jst-train_pkl')
test_dataset = sa_dataloader.SaDataset('jst-test_pkl')


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False)


# Fully connected neural network with one hidden layer
# class NeuralNet(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(NeuralNet, self).__init__()
#         self.bn1 = nn.BatchNorm1d(input_size)
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.bn2 = nn.BatchNorm1d(hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.bn3 = nn.BatchNorm1d(hidden_size)
#         self.fc3 = nn.Linear(hidden_size, num_classes)
#
#     def forward(self, x):
#         out = self.fc1(self.bn1(x))
#         out = self.bn2(torch.tanh(out))
#         out = self.fc2(out)
#         out = self.bn3(torch.tanh(out))
#         out = self.fc3(out)
#         return out

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        return out


model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
model.train()
for epoch in range(num_epochs):
    for i, (x, y) in enumerate(train_loader):
        # Move tensors to the configured device
        x = x.float().to(device)
        y = y.float().to(device)
        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
model.eval()
with torch.no_grad():
    loss_list = []
    for x, y in test_loader:
        x = x.float().to(device)
        y = y.float().to(device)
        outputs = model(x)
        loss = criterion(outputs.view(-1), y)
        loss_list.append(loss)
        # total += y.size(0)
        # correct += (predicted == labels).sum().item()

    print('loss of the network: {}'.format(sum(loss_list)/len(loss_list)))

# Save the model checkpoint
torch.save(model.state_dict(), './model.ckpt')


