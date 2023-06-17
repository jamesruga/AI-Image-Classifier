# IMPORTS ____________________________________________________________________________________________________________________________________________________

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

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
parser = argparse.ArgumentParser(description = " Reads in an image and a checkpoint then prints the most likely image class and it's associated probability")

#Add parser arguments
parser.add_argument('image_path', action='store', nargs = '?', default = 'flowers/test/1/image_06743.jpg',  help= "add the path data to the image to be predicted: 'path/to/image'")
parser.add_argument('--checkpoint', action='store', nargs = '?', default = 'checkpoint.pth', help= "add directory for loading model checkpoint: 'path/to/checkpoint'")
parser.add_argument('--top_k', action="store", default=5, type=int, help="enter number of K for top K most likely classes")
parser.add_argument('--category_names', action="store", type = str, default="cat_to_name.json", help="get path to mapping of category to name json file")
parser.add_argument('--gpu', action = 'store_true', default = False, help = "train using GPU")

# create variable to store parsed arguments
args = parser.parse_args()




# DATA SECTION __________________________________________________________________________________________________________________________________________________________________

# Directories
checkpoint_dir = args.checkpoint
image_path = args.image_path
category_names = args.category_names

# Defining the label mapper
import json

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
 




# IMAGE PROCESSING ___________________________________________________________________________________________________________________________________________


# Processing a PIL image for use in a PyTorch model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    
    image = Image.open(image)
    image.load()
        
    # Scale image
    width, height = image.size
    if width < height:
        image.thumbnail((256, 256 * height // width))
    else:
        image.thumbnail((256 * width // height, 256))
    
    # Crop
    width, height = image.size
    left = (width - 224) // 2
    top = (height - 224) // 2
    right = (width + 224) // 2
    bottom = (height + 224) // 2
    image = image.crop((left, top, right, bottom))
    
    # Convert image to a numpy array
    np_image = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Transpose the color channel to the first dimension
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image


# Fn to return a plotted image
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax






# PREDICTION _________________________________________________________________________________________________________________________________________________


#Processing device selection

if args.gpu and torch.cuda.is_available(): 
    device = torch.device('cuda')
elif args.gpu and not torch.cuda.is_available():
    device = torch.device('cpu')
    print('CPU will be used because GPU is unavailable!!')
else:
    device = torch.device('cpu')
    
    
    
    
# Loading checkpoint and rebuilding the model

def load_checkpoint(checkpoint_dir):
    
    model = models.vgg16(pretrained=True)
    optimizer = optim.Adam(model.classifier.parameters(), lr= 0.03)

    checkpoint = torch.load(checkpoint_dir)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx_mapping']
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    
   
    hidden_layer1 = checkpoint['hidden_layer1']
    hidden_layer2 = checkpoint['hidden_layer2']
    epochs = checkpoint['epochs']
    
    return model, optimizer, epochs, hidden_layer1, hidden_layer2


model, optimizer, epochs, hidden_layer1, hidden_layer2 = load_checkpoint(checkpoint_dir)

# Implementing the code to predict the class from an image file
def predict(image_path, model, topk = args.top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
   
    model.to(device)
    model.eval()
    
    # Preprocessing the image
    img = process_image(image_path)
   
    
    # Convert image to a tensor
    tensor_img = torch.from_numpy(img).type(torch.FloatTensor)
    
    # Add a batch dimension to the image
    processed_img = tensor_img.unsqueeze_(0)
    processed_img = torch.tensor(processed_img)
    # Forward pass
    image = processed_img.to(device)
    output = model.forward(image)
    
    # Calculate the class probabilities
    probs = torch.exp(output)
    
    # Get the topk largest values and their indices
    top_probs, top_labels = probs.topk(topk)
       
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[label] for label in top_labels.cpu().numpy()[0]]
    
    
    return top_probs.detach().cpu().numpy()[0], top_labels


# Get the top 5 classes and corresponding probabilities

top_probs, top_labels = predict(image_path, model)

# Convert the labels to strings
top_labels = [cat_to_name[label] for label in top_labels]


# Printing prediction results

print('\n\n\n Image given: {}\n'.format(image_path))
print('Predicted as: \n')
print(top_labels)
print('\n...with the following probabilities respectively : \n')
print(top_probs)


# Displaying an image plot along with the top 5 classes

def sanity_checker(image_path, mapper):
    # Get the top 5 classes and corresponding probabilities

    top_probs, top_labels = predict(image_path, model)

    # Convert the labels to strings
    top_labels = [mapper[label] for label in top_labels]

    print(top_probs)
    print(top_labels)
    
    # Get the file name without the extension
    filename =  image_path.split('/')[-2]
    
    # Remove the file extension from the file name
    flower_title = mapper[filename]

    # Load the image
    image = Image.open(image_path)
    

    # Convert the image to a NumPy array
    image = np.array(image)

    # Invert the image
    image[:,:,0] = 255 - image[:,:,0]  # Invert the red channel
    image[:,:,1] = 255 - image[:,:,1]  # Invert the green channel
    image[:,:,2] = 255 - image[:,:,2]  # Invert the blue channel


    # Convert the NumPy array to a PyTorch tensor
    image = torch.from_numpy(image).type(torch.FloatTensor)

    # Make the plot
    fig, (ax1, ax2) = plt.subplots(figsize=(6,10), nrows=2)

    # Display the image
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title(flower_title)

    # Plot the bar graph
    y_pos = np.arange(len(top_probs))
    ax2.barh(y_pos, top_probs, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_labels)
    ax2.invert_yaxis()
    ax2.set_title('Class Probability')

    plt.tight_layout()
    plt.show()
    
    return

# Plot results
sanity_checker(image_path, cat_to_name)

      
      

# Thanks to Zahraa Al-Sahili - My Session Lead at Udacity AIPND Nanodegree Scholarship Program 
# https://katba-caroline.com/wp-content/uploads/2018/11/Image-Classifier-Project.html
# https://github.com/rebeccaebarnes/DSND-Project-2
# https://docs.python.org/3/library/argparse.html
# https://pillow.readthedocs.io/en/latest/reference/Image.html
# https://github.com/ErkanHatipoglu/AIPND_final_project_part_2
# AWS AIPND Nanodegree Scholarship in collaboration with Udacity and Intel