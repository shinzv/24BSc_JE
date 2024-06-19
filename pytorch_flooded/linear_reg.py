import matplotlib.pyplot as plt
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split, default_collate

torch.manual_seed(400)

#Training parameters
flood_level = 0
learning_rate = 0.001
data_amount = 500
batch_size = 50
epochs = 100

split = 0.8
training_amount = int(data_amount * 0.8)

#Creating the dataset
class Data(Dataset):
    def __init__(self):
        self.x = torch.zeros(data_amount, 5)
        self.x[:, 0] = torch.randn(data_amount)
        self.x[:, 1] = torch.randn((self.x.shape[0]))
        self.x[:, 2] = torch.randn((self.x.shape[0]))
        self.x[:, 3] = torch.randn((self.x.shape[0])) 
        self.x[:, 4] = torch.randn((self.x.shape[0]))
        #Target values
        self.y = torch.mul(self.x[:, 0], 1) + torch.mul(self.x[:, 1], 1) +  torch.mul(self.x[:, 2], 1) + torch.mul(self.x[:, 3], 1)  + torch.mul(self.x[:, 4], 1) + 0
        self.len = self.x.shape[0]

    def __getitem__(self, idx):          
        return self.x[idx], self.y[idx] 

    def __len__(self):
        return self.len

#Creating a dataset object
data_set = Data()
train, test = random_split(data_set, [training_amount, data_amount - training_amount])

#Creating a custom Multiple Linear Regression Model
class MultipleLinearRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultipleLinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    #Makes a prediction
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

#Creating the model object
MLR_model = MultipleLinearRegression(5,1)

#defining the model optimizer
optimizer = torch.optim.SGD(MLR_model.parameters(), lr=learning_rate)

#flooded loss criterion
def criterion(y_pred, y):
    return (abs(torch.mean((y_pred - y) ** 2) - flood_level) + flood_level)

#mse criterion for tracking of the loss
def criterion_mse(y_pred, y):
    return torch.mean((y_pred - y) ** 2)

#Creating the dataloader
train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle = True)
train_loader_all = DataLoader(dataset=train, batch_size=training_amount)

#Tracking all important values
losses = []
losses_test = []

x_test, y_test = default_collate(test)

#Training loop
for epoch in range(epochs):
    #Training of the model
    for x,y in train_loader:
        y_pred = MLR_model(x)

        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   

    #Tracking of results
    for x2,y2 in train_loader_all:
        y_pred_all = MLR_model(x2)
        y_pred_test = MLR_model(x_test)
        y_all = y2
        tracked_loss = criterion_mse(y_pred_all, y_all)
        tracked_loss_test = criterion_mse(y_pred_test, y_test)
        losses.append(tracked_loss.item())
        losses_test.append(tracked_loss_test.item())

#Plotting the losses
plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(6.5,4.5))
plt.plot(losses, color = "blue", label="Training loss")
plt.plot(losses_test, color = "red", label="Testing loss")
plt.title("")
plt.xlabel("Iterations", labelpad=10)
plt.ylabel("Loss",labelpad=10)
plt.legend(fontsize=12, loc=1)
plt.subplots_adjust(bottom=0.15, left = 0.14, right = 0.92)
plt.show()



