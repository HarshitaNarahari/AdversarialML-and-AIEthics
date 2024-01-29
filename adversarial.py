import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils import *
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

if torch.backends.mps.is_available():
	device=torch.device("mps")
elif torch.cuda.is_available():
	device=torch.device("cuda")
else:
	device=torch.device("cpu")

print(device)

# define CNN for a 3-class problem with input size 160x160 images
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
		self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
		self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
		self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
		self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(256 * 5 * 5, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, 3)
		self.relu = nn.ReLU()
		self.final_activation = nn.LogSoftmax(dim=1)

	def forward(self, x):
		x = self.pool(self.relu(self.conv1(x)))
		x = self.pool(self.relu(self.conv2(x)))
		x = self.pool(self.relu(self.conv3(x)))
		x = self.pool(self.relu(self.conv4(x)))
		x = self.pool(self.relu(self.conv5(x)))
		x = x.view(-1, 256 * 5 * 5)
		x = self.relu(self.fc1(x))
		x = self.relu(self.fc2(x))
		x = self.fc3(x)
		x = self.final_activation(x)
		return x



# Load dataset
train_dir = './data/train'
test_dir = './data/test'
image_size = 160
batch_size = 16
workers = 0

class CropToSmallerDimension(transforms.Resize):
    def __init__(self, size, interpolation=Image.BILINEAR):
        super().__init__(size, interpolation)

    def __call__(self, img):
        # Get the original image size
        width, height = img.size

        # Determine the smaller dimension
        smaller_dimension = min(width, height)

        # Crop the image to the smaller dimension
        return transforms.CenterCrop(smaller_dimension)(img)

train_dataset = datasets.ImageFolder(root=train_dir, transform=transforms.Compose([CropToSmallerDimension(256),transforms.ToTensor(),transforms.Resize(image_size)]))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

test_dataset = datasets.ImageFolder(root=test_dir, transform=transforms.Compose([CropToSmallerDimension(256),transforms.ToTensor(),transforms.Resize(image_size)]))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

print('Number of training images: {}'.format(len(train_dataset)))
print('Number of test images: {}'.format(len(test_dataset)))
print('Detected Classes are: ', train_dataset.classes) # classes are detected by folder structure

# Define the attack
# some of the code below is from corrections using ChatGPT and adversarial-ml-tutorial.org
def FGSM(model, image, label, epsilon):
	#evaluates the model
	model.eval()

	# image and label on same device as model 
	# allows gradient computation using image.requires_grad
	image, label= image.to(device), label.to(device)
	image.requires_grad= True

	#forward pass to image and defines the loss function
	output = model(image)
	loss = F.nll_loss(output, label)

	# backward pass to gradient for calculating
	model.zero_grad()
	#loss function
	loss.backward()

	#gets data from grad on the image
	data_grad = image.grad.data

	# gets the element-wise sign of data gradient
	# creates the pertrubed_image using the FGSM formula
	perturbed_image = image + epsilon * data_grad.sign()

	#crops the perturbed_image to stay within the pixel range specified
	perturbed_image = torch.clamp(perturbed_image, 0, 1)

	return perturbed_image



# some of the code below is from corrections using ChatGPT and adversarial-ml-tutorial.org
def PGD(model, image, label, epsilon, iterations, alpha):
	#evaluates the model
	model.eval()

	# image and label on same device as model 
	# allows gradient computation using image.requires_grad
	image, label= image.to(device), label.to(device)
	image.requires_grad= True

	# creates a clone of the original before making it a perturbed_image
	perturbed_image = image.clone().detach().requires_grad_()

	# recurses through the iterations(number unknown)
	for _ in range(iterations):

		#forward pass to perturbed_image
		output = model(perturbed_image)
		loss = F.nll_loss(output, label)

		# backward pass to gradient for calculating
		model.zero_grad()
		#loss function
		loss.backward()

		#gets data from grad on the perturbed_image
		data_grad = perturbed_image.grad.data

		# gets the element-wise sign of data gradient
		# creates the pertrubed_image using the PGD formula
		perturbed_image = perturbed_image + alpha * data_grad.sign()

		# projects perturbed_image on ε-ball and updates the values accordingly
		perturbed_image.data = torch.max(torch.min(perturbed_image, image + epsilon), image - epsilon)

		#crops the perturbed_image to stay within the pixel range specified
		perturbed_image = torch.clamp(perturbed_image, 0, 1)

		#detach the perturbed_image and reattch the perturbed_image with the requires_grad_ function
		perturbed_image = perturbed_image.detach().requires_grad_()

	return perturbed_image



net = Net()
net.to(device)

# Train the network

# criterion = nn.NLLLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.001)
# epochs = 100
# running_loss = 0
# train_losses, test_losses = [], []
# i=0

# for epoch in tqdm(range(epochs)):
# 	for inputs, labels in train_dataloader:
# 		inputs, labels = inputs.to(device), labels.to(device)
# 		optimizer.zero_grad()
# 		logps = net(inputs)
# 		loss = criterion(logps, labels)
# 		loss.backward()
# 		optimizer.step()
# 		running_loss += loss.item()

# # Save the model
# torch.save(net.state_dict(), 'model.pth')


# Test the model
net.load_state_dict(torch.load('model.pth', map_location="cpu"))
net.to(device)

correct=[]

net.eval()
accuracy = 0
for inputs, labels in tqdm(test_dataloader):
	inputs, labels = inputs.to(device), labels.to(device)
	outputs = net(inputs)
	_, predicted = torch.max(outputs.data, 1)
	accuracy += (predicted == labels).sum().item()
	correct.append((predicted == labels).tolist())

print('Accuracy of the network on the test images: %d %%' % (100 * accuracy / len(test_dataset)))


# Test the model with adversarial examples
# Save adversarial examples for each class using FGSM with (eps = 0.01, 0.05, 0.1)
# Save one adversarial example for each class using PGD with (eps = 0.01, 0.05, 0.1, alpha = 0.001, 0.005, 0.01 respectively, iterations = 20)

# some of the code below is from corrections using ChatGPT and adversarial-ml-tutorial.org

#fgsm
fgsm_alphas = [0.001, 0.01, 0.1]

# pgd 
pgd_alpha = 0.00784313725 #2/255
pgd_iterations = 50
pgd_eps = [0.01, 0.05, 0.1]


# recurses through the fgsm_alphas values
for alpha in fgsm_alphas:
	#sets the correct and total sum variables to 0
	correct = 0
	total = 0

	# recurses through the images, labels in the test_dataloader
	for data in test_dataloader:
		# gets images and labels from data
		# image and label on same device as model 
		images, labels = data
		images, labels = images.to(device), labels.to(device)

		# generates the adversarial examples using the fgsm formula on the net model
		perturbed_images = FGSM(net, images, labels, alpha)

		# receives the outputs of the net model on the perturbed_images
		outputs = net(perturbed_images)

		#receives the predicted values of the net model
		_, predicted = torch.max(outputs.data, 1)

		#adds the labels size to total to get the total values
		total += labels.size(0)

		#adds the amount of predicted values that match the labels to correct
		correct += (predicted == labels).sum().item()

	#calculates accuracy by dividing the correct values by total values
	accuracy = correct/total

	print(f'Accuracy of the network on FGSM adversarial test images (alpha value={alpha}): {accuracy}')


# recurses through the pgd epsilon values
for eps in pgd_eps:
	#sets the correct and total sum variables to 0 again
	correct = 0
	total = 0

	# recurses through the data in the test_dataloader
	for data in test_dataloader:
		# gets images, labels from data
		# image and label on same device as model 
		images, labels = data
		images, labels = images.to(device), labels.to(device)

		# generates the adversarial examples using the pgd formula on the net model
		perturbed_images = PGD(net, images, labels, eps, pgd_iterations, pgd_alpha)

		# receives the output of the net model
		outputs = net(perturbed_images)

		#receives the predicted values of the net model
		_, predicted = torch.max(outputs.data, 1)

		#adds the labels size to total to get the all values
		total += labels.size(0)

		#adds the amount of predicted values that match the labels to correct
		correct += (predicted == labels).sum().item()

	#calculates accuracy by dividing the correct values by total values
	accuracy = correct/total

	print(f'Accuracy of the network on PGD adversarial test images (epsilon value={eps}): {accuracy}')

# Original
# print('Accuracy of the network on adversarial test images: %d %%' % (100 * accuracy / len(test_dataset)))


# function to provide the image - used chatgpt to fix my errors for the functions below
def imshow(img, title):
	#makes numpy image
	npimg = img.numpy()
	#transposes the image
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	# displays title of image
	plt.title(title)
	plt.show()

#assuming we're getting one test example from each class - 40 indices
indices = [40]

#recurse through the 40 indices 
for i in indices:
	#retrieves original_image and original_label
	original_image, original_label = test_dataset[i]
	#adds the batch dimension to the original
	original_image = original_image.unsqueeze(0)

	#displays the original image with its specified class
	imshow(vutils.make_grid(original_image, normalize= True), f'Original Image (Class: {original_label})')

	#generates the adversial examples for the alpha values
	fgsm_alphas = [0.001, 0.01, 0.1]

	#recurses through the fgsm alphas
	for alpha in fgsm_alphas:
		# generates the adversial examples for the alpha values using the fgsm formula
		fgsm_perturbed_image = FGSM(net, original_image, torch.tensor([original_label]), alpha)

		# receives the output of the net model
		outputs = net(fgsm_perturbed_image)

		#receives the predicted class of the net model
		_, predicted_class = torch.max(outputs.data, 1)

		# configures the title for the image
		title = f'FGSM adversarial adversarial image for α={alpha}: Predicted as {train_dataset.classes[predicted_class.item()]}'
		
		# displays an image example of the adversarial image
		imshow(vutils.make_grid(fgsm_perturbed_image, normalize=True), title)
	
	# assigns the pgs epsilons
	pgd_epsilons = [0.001, 0.01, 0.1]

	# recurses through the pgd epsilons
	for eps in pgd_epsilons:
		# generates the adversial examples for the alpha values using the pgd formula
		pgd_perturbed_image = PGD(net, original_image, torch.tensor([original_label]), eps, pgd_iterations, pgd_alpha)

		# receives the output of the net model
		outputs = net(pgd_perturbed_image)

		#receives the predicted class of the net model
		_, predicted_class = torch.max(outputs.data, 1)

		# configures the title for the image
		title = f'PGD adversarial image for ε={eps}: Predicted as {train_dataset.classes[predicted_class.item()]}'

		# displays an example of the adversarial image
		imshow(vutils.make_grid(pgd_perturbed_image, normalize=True), title)


# Additional 10 epochs of training with PGD adversarial examples
adam_learning_rate = 0.001
optimizer = optim.Adam(net.parameters(), lr= adam_learning_rate)

#pgd values for alpha, iterations and epsilon
pgd_alpha = 0.00784313725 #2/255
pgd_iterations = 50
pgd_eps = 0.075

#recurses 10 epochs
for epoch in tqdm(range(10)):
	#sets the running loss to start at 0 before iterations
    epoch_running_loss = 0.0
	
    for inputs, labels in tqdm(test_dataloader):
		# inputs and labels on the same device as model
    	# allows gradient computation using input.requires_grad
        inputs, labels = inputs.to(device), labels.to(device)
		
        # Generates PGD adversarial examples
        p_inputs = PGD(net, inputs, labels, pgd_eps, pgd_iterations, pgd_alpha)

        # backward pass to gradient for calculating
        optimizer.zero_grad()

        # forward pass to perturbed image inputs
        outputs = net(p_inputs)
        loss = F.nll_loss(outputs, labels)

        # backward pass to gradient for calculating
		# loss function
        loss.backward()
        optimizer.step()

		# adds the current epoch's loss to the total running loss
        epoch_running_loss += loss.item()

    # calculates the average epoch loss
    print(f'Epoch #{epoch+1}, Average Epoch Loss: {epoch_running_loss / len(test_dataloader)}')

#evaluates the models
net.eval()

#cleaned images variables
correct_cleaned = 0
total_cleaned = 0

#recurses through the inputs, labels in test_dataloader
for inputs, labels in tqdm(test_dataloader):
	# inputs and labels on the same device as model
	inputs, labels = inputs.to(device), labels.to(device)

	#outputs of net model
	outputs = net(inputs)

	#retrieved predicted values from outputs of data
	_, predicted = torch.max(outputs.data, 1)

	# adds the labels length to total
	# adds the correct predictions to corrected
	total_cleaned += labels.size(0)
	correct_cleaned += (predicted == labels).sum().item()

# calculates the clean images accuracy
accuracy_cleaned = correct_cleaned/total_cleaned
print(f'Cleaned test images accuracy:{accuracy_cleaned}')

#cleaned images variables
correct_cleaned_pgd = 0
total_cleaned_pgd = 0

#recurses through the data in test_dataloader
for data in test_dataloader:
	# retrieved inputs and labels from data
	images, labels = data
	# inputs and labels on the same device as model
	images, labels = images.to(device), labels.to(device)

	#generates the pgd adversarial examples using the pgd formula for a perturbed_images
	perturbed_images_pgd = PGD(net, images, labels, pgd_eps, pgd_iterations, pgd_alpha)

	#retrieves outputs from net model
	#retrieves predicted from the outputs
	outputs_pgd = net(perturbed_images_pgd)
	_, predicted_pgd = torch.max(outputs_pgd.data, 1)

	# adds the labels length to pgd total
	# adds the correct predictions to pgd corrected
	total_cleaned_pgd += labels.size(0)
	correct_cleaned_pgd += (predicted_pgd == labels).sum().item()

# calculates the pgd images accuracy
accuracy_cleaned_pgd = correct_cleaned_pgd/total_cleaned_pgd
print(f'PGD adversarial test images accuracy:{accuracy_cleaned_pgd}')





















