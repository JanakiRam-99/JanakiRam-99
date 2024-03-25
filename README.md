import torchvision.transforms as transforms 

from torchvision import datasets 

from torch.utils.data import DataLoader 

import matplotlib.pyplot as plt 

# loading training data train_dataset=datasets.MNIST(root='./data',train=True, 

transform=transforms.ToTensor(),download=True) 

# loading test data 

test_dataset = 

datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor()) 

print("number of training samples: " + str(len(train_dataset)) + "\n" + "number of 

testing samples: " + str(len(test_dataset)))

print("datatype of the 1st training sample: ", train_dataset[0][0].type()) 

print("size of the 1st training sample: ", train_dataset[0][0].size()) 

print("label of the first taining sample: ", train_dataset[0][1]) 

print("label of the second taining sample: ", train_dataset[1][1]) img_5 = 

train_dataset[0][0].numpy().reshape(28, 28) 

plt.imshow(img_5, cmap='gray') 

plt.show() img_0 = train_dataset[1][0].numpy().reshape(28, 28) 

plt.imshow(img_0, cmap='gray') 

plt.show() batach_size = 32 train_loader = 

DataLoader(dataset=train_dataset, batch_size=batach_size, shuffle=True) 

test_loader = DataLoader(dataset=test_dataset, batch_size=batach_size, 

shuffle=False) 

class LogisticRegression(torch.nn.Module): \

def __init__(self, n_inputs, n_outputs): 

 super().__init__() 

self.linear = torch.nn.Linear(n_inputs, n_outputs) 

def forward(self, x): 

 y_pred = torch.sigmoid(self.linear(x)) 

return y_pred n_inputs = 28*28 

# makes a 1D vector of 784 

n_outputs = 10 log_regr = 

LogisticRegression(n_inputs, n_outputs)

optimizertorch.optim.SGD(log_regr.parameters(),lr

=0.001) 

criterion = torch.nn.CrossEntropyLoss() 

epochs = 50 

Loss = [] 

acc = [] for epoch in range(epochs): 

for i, (images, labels) in enumerate(train_loader): 

optimizer.zero_grad() 

outputs = log_regr(images.view(-1, 28*28)) 

loss = criterion(outputs, labels) 

# Loss.append(loss.item()) 

loss.backward() 

optimizer.step() 

Loss.append(loss.item()) 

correct = 0 

for images, labels in test_loader: 

 outputs = log_regr(images.view(-1, 28*28)) 

_, predicted = torch.max(outputs.data, 1) 

correct += (predicted == labels).sum() 

accuracy = 100 * (correct.item()) / len(test_dataset) 

acc.append(accuracy)

print('Epoch: {}. Loss: {}. Accuracy: {}'.format(epoch, loss.item(), accuracy)) 

plt.plot(Loss) 

plt.xlabel("no. of epochs") 

plt.ylabel("total loss") 

plt.title("Loss") 

plt.show() 

plt.plot(acc) 

plt.xlabel("no. of epochs") 

plt.ylabel("total accuracy") 

plt.title("Accuracy") 

plt.show() 

print("label of the first testing sample: ", test_dataset[0][1]) 

print("label of the second testing sample: ", test_dataset[1][1]) img_7 = 

test_dataset[0][0].numpy().reshape(28, 28) 

plt.imshow(img_7,cmap='gray') p

lt.show() 

img_2 = test_dataset[1][0].numpy().reshape(28, 28) 

plt.imshow(img_2, cmap='gray') plt.show() 

#Prediction 1 

sample_index = 5 

sample_image, sample_label = test_dataset[sample_index]
output = log_regr(sample_image.view(-1, 28*28)) 

plt.imshow(sample_image, cmap='gray') plt.title(f"Actual Label: 

{sample_label}, Predicted Label: {predicted_class.item()}")

plt.show()

sample_index = 10 

sample_image, sample_label = test_dataset[sample_index]

output = log_regr(sample_image.view(-1, 28*28)) 

_, predicted_class = torch.max(output, 1) 

print(f"Actual Label: {sample_label}, PredictedLabel: 

{predicted_class.item()}")

sample_image = sample_image.numpy().reshape(28, 28)

plt.imshow(sample_image, cmap='gray')

plt.title(f"Actual Label: 

{sample_label}, Predicted {predicted_class.item()}")

plt.show()
