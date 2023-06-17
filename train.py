# IMPORTS ____________________________________________________________________________________________________________________________________________________

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import torch
import time
import os
import argparse

from PIL import Image
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models 


# COMMAND LINE SECTION __________________________________________________________________________________________________________________________________________________________

# Create parser object
parser = argparse.ArgumentParser(description = 'Train a network on a dataset')

#Add parser arguments
parser.add_argument('data_dir', action='store', type = str, default = 'flowers',  help= 'add the main data directory for the image set')
parser.add_argument('--save_dir', action='store', type = str, default = 'checkpoint.pth', help= 'add directory for saving model checkpoint')
parser.add_argument('--arch',
                    choices =['vgg16','vgg13','vgg19'], default = 'vgg16',
                    help= 'add pre-trained model type to accelerate training process. Options are: vgg13, vgg16 or vgg 19')
parser.add_argument('--learning_rate', type=float, default = 0.003, help = "learning rate")
parser.add_argument('--hidden_units', type=int, default = [2048, 1024], help = "number of hidden layers units for two layers. Format like [int, int]")
parser.add_argument('--epochs', type=int, default = 10, help = "number of training iterations(epochs)")
parser.add_argument('--gpu', action = 'store_true', default = False, help = "train using GPU")

# create variable to store parsed arguments
args = parser.parse_args()


# DATA SECTION __________________________________________________________________________________________________________________________________________________________________

# Directories
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
save_dir = args.save_dir

# Transforms for the training, validation and testing datasets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
# Loading the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

# Defining dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)


# Defining the label mapper
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
 




# BUILDING AND TRAINING NETWORK _________________________________________________________________________________________________________________________________________________

#Processing device selection

if args.gpu and torch.cuda.is_available(): 
    device = torch.device('cuda')
elif args.gpu and not torch.cuda.is_available():
    device = torch.device('cpu')
    print('CPU will be used because GPU is unavailable!!')
else:
    device = torch.device('cpu')

    

#Loading pretrained model

arg_architecture = args.arch

if arg_architecture == 'vgg13':
    model = models.vgg13(pretrained=True)
    print('VGG13')
    print(model.classifier)
    
elif arg_architecture == 'vgg16':
    model = models.vgg16(pretrained=True)
    print('VGG16')
    print(model.classifier)
else:
    model = models.vgg19(pretrained=True)
    print('VGG19')
    print(model.classifier)

# Freezing pre_trained parameters

for param in model.parameters():
    param.requires_grad = False
    

    
#Building model

model.classifier = nn.Sequential(nn.Linear(25088, args.hidden_units[0]),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(args.hidden_units[0], 
                                           args.hidden_units[1]),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(args.hidden_units[1], 102),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

model.to(device);


#Training and validation

print('Training in progress.......... Please wait!')
print()
print('Training on GPU? \n{}'.format(args.gpu))
print('Warning! Training might take longer if on CPU!')
print()
print('Displaying training loss, validation loss and accuracy.... ')

epochs = args.epochs
steps = 0
running_loss = 0
print_every = 5

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()
       

 
# TESTING ____________________________________________________________________________________________________________________________________________________          
# Testing validation on test set  

test_loss = 0
accuracy = 0
model.eval()
        
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)
                    
        test_loss += batch_loss.item()
                    
        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

print('Displaying test loss and accuracy on test dataset...\n')
print(f"Test loss: {test_loss/len(testloader):.3f}.. "
      f"Test accuracy: {100 * accuracy/len(testloader):.3f} %")



# SAVING AND LOADING MODEL ______________________________________________________________________________________________________________________________________________________

# Saving the checkpoint

print(model.classifier)
model.class_to_idx = train_data.class_to_idx

checkpoint = {'hidden_layer1': args.hidden_units[0],
              'hidden_layer2': args.hidden_units[1],
              'epochs': epochs,
              'class_to_idx_mapping': model.class_to_idx,
              'optimizer_state': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'classifier': model.classifier
              }

torch.save(checkpoint, save_dir)

print('\n\n\n Model saved!\n\n\n')



# Loading checkpoint and rebuilding the model

def load_checkpoint(save_dir):
    
    checkpoint = torch.load(save_dir)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    
   
    hidden_layer1 = checkpoint['hidden_layer1']
    hidden_layer2 = checkpoint['hidden_layer2']
    epochs = checkpoint['epochs']
    
    return model, optimizer, epochs, hidden_layer1, hidden_layer2


reloaded_model, reloaded_optimizer, epochs, h1, h2 = load_checkpoint(save_dir)

print('Saved model: \n {}'.format(reloaded_model.classifier))
print('Saved optimizer: \n {}'.format(reloaded_optimizer))
print('Hidden layer units: [{}, {}]'.format(h1, h2))
print('Number of epochs used: {} '.format(epochs))




# Thanks to Zahraa Al-Sahili - my Session Lead at Udacity AIPND Nanodegree Scholarship Program 
# https://katba-caroline.com/wp-content/uploads/2018/11/Image-Classifier-Project.html
# https://github.com/rebeccaebarnes/DSND-Project-2
# https://docs.python.org/3/library/argparse.html#nargs
# https://pillow.readthedocs.io/en/latest/reference/Image.html
# https://github.com/ErkanHatipoglu/AIPND_final_project_part_2
# AWS AIPND Nanodegree Scholarship in collaboration with Udacity and Intel