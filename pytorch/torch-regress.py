#learn-pytorch-through-linear-regression
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#1. Set random seed for reprodcibility
torch.manual_seed(42)

#2. Generate synthetic data
#Assume the true relattionship is y = 2x + 1 + noise
num_samples = 100
X = torch.randn(num_samples, 1)
true_weight = 2.0
true_bias = 1.0
y = true_weight * X + true_bias + torch.randn(num_samples,1) * 0.1  # Add small noise

# Define the linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        #nn.Linear(input_dimension, output_dimension)
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        return self.linear(x)
    
#Instantiate the model
model = LinearRegressionModel()

#4. Define loss function and optimiazer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
#Stochastic Gradient Descent, learning rate 0.01

#5. Train the model
num_epochs = 100
losses = []

for epoch in range(num_epochs):
    #Forward pass: Computer predicted y by passing x to the model
    predictions = model(X)

    #Compute loss
    loss = criterion(predictions, y)
    losses.append(loss.item())

    #Backward pass: Compute gradients
    optimizer.zero_grad()
    loss.backward()
    
    #Update parameters
    optimizer.step()

    if(epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss:{loss.item():.4f}')

#6. Get trained parameters
[w,b] = model.linear.parameters()
print(f'\nTrue Weight:{true_weight},True Bias:{true_bias}')
print(f'Learn Weight: {w.item():.4f},Learn Bias:{b.item():.4f}')

#7. Visualize results
plt.figure(figsize=(10,6))

#Plot fitted line , detach from computation graph and convert to numpy array
predicted_y = model(X).detach().numpy()
plt.plot(X.numpy(), predicted_y,color='red',linewidth=2, label='Fitted Line')

plt.title("Pytorch Linear Regression Result")
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

#Optional: Plot loss curver over epochs
plt.figure(figsize=(10,4))
plt.plot(range(num_epochs),losses,color='green')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()