import os 
import numpy as np 
import pandas as pd 
from PIL import Image as img
import torch 
from torch.utils.data import Dataset, DataLoader 
import torchvision.transforms as transforms 
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Path to dataset
dataset_path = 'Desktop/DSC3/Project/Dataset/'
categories = ['malignant', 'normal']

image_paths = []
labels = []

#Load and Preprocess Data 
for category in categories:
    class_path = os.path.join(dataset_path, category)
    for filename in os.listdir(class_path):
        if not filename.startswith('.'): #added this line to skip any hidden files that gave me errors
            img_path = os.path.join(class_path, filename)
            if img_path.lower().endswith('.png'): #only load images ending with .png
                image_paths.append(img_path)
                labels.append(1 if category == 'malignant' else 0)

class MedicalImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform = None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform 
    
    def __len__(self): 
        return len(self.image_paths)

    def __getitem__(self, idx): 
        image_path = self.image_paths[idx]
        image = img.open(image_path)
        image = img.open(image_path).convert("RGB") #convert images to rgb

        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label 

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = MedicalImageDataset(image_paths, labels, transform=transform)

from sklearn.model_selection import train_test_split

#split data into training and remaining data
train_paths, remaining_paths, train_labels, remaining_labels = train_test_split(
    image_paths, labels, test_size=0.3, random_state=42)

#split the remaining data into validation and test
validation_split = 0.5  # Split the remaining data evenly
val_paths, test_paths, val_labels, test_labels = train_test_split(remaining_paths, 
                                                                  remaining_labels, test_size=validation_split, random_state=42)

#create dataset for training, validation and testing split
train_dataset = MedicalImageDataset(train_paths, train_labels, transform=transform)
val_dataset = MedicalImageDataset(val_paths, val_labels, transform=transform)
test_dataset = MedicalImageDataset(test_paths, test_labels, transform=transform)

#initialize Data Loaders for each dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#Define CNN architecture 
import torch.nn as nn
import torch.nn.functional as F  

class SimpleCNN(nn.Module): 
    def __init__(self): 
        super(SimpleCNN, self).__init__()
        #Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1) 

        #Max pooling layer 
        self.pool= nn.MaxPool2d(2, 2)

        #Fully Connected Layers 
        self.fc1 = nn.Linear(64 * 28 * 28, 512) 
        self.fc2 = nn.Linear (512, 2)

        #Dropout Layer (p = 0.5) 
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x): 
        #Sequence of convolutional and max pooling layers 
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        #Flatten image input 
        x = x.view(-1, 64 * 28 * 28)
        #Dropout layer 
        x = self.dropout(x)
        #hidden layer with relu activation function 
        x = F.relu(self.fc1(x))
        #Dropout layer 
        x = self.dropout(x)
        #hidden layer 
        x = self.fc2(x)
        return x
    
model = SimpleCNN()

#Train the model 
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

num_epochs = 10 

#Training loop
for epoch in range(num_epochs):
    model.train()
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        outputs = model(images) #Forward pass   
        loss = criterion(outputs, labels) 
        optimizer.zero_grad() #Backward pass & optimization
        loss.backward()
        optimizer.step()
   

    #evaluation step on test set 
    model.eval()
    with torch.no_grad(): 
        correct = 0
        total = 0
        for images, labels in test_loader: 
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = 100 * correct / total 

   # test_accuracies.append(test_accuracy)
    print(f'Accuracy on test set: {test_accuracy}%')

    correct = [] #empty array to store true labels
    predicted = [] #empty array to store predicted labels 

    for images, labels in test_loader: 
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct.extend(labels.numpy()) #populate empty array
        predicted.extend(preds.numpy()) #populate empty array

    # Calculate the confusion matrix for the test set
    conf_mat_test = confusion_matrix(correct, predicted)


print(conf_mat_test)
print(classification_report(correct, predicted, digits=4)) #classification report as another measure of model efficiency / accuracy
torch.save(model.state_dict(), 'model.pth') #save state dictionary of model after training loop 

        

