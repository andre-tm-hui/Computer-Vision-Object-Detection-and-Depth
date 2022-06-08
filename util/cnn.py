import numpy as np
import cv2
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
import os;
from torch.utils.data import Dataset, DataLoader

datadir = 'data/train'

# set image transformations to be applied to datasets
data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

trainset = datasets.ImageFolder("data/train", transform = data_transform)

trainloader = DataLoader(trainset, batch_size = 4, shuffle = True, num_workers = 4)

testset = datasets.ImageFolder("data/test", transform = data_transform)

testloader = DataLoader(testset, batch_size = 4, shuffle = True, num_workers = 4)


def trainCNN():
	train = True			# set to false if you just want to see the accuracy data for the CNN

	# check CUDA availability
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# initialise Resnet model
	net = torchvision.models.wide_resnet50_2()
	PATH = './util/handModel_5g.pth'

	# load pre-existing weights - set the second variable depending on whether you want to train a
	# a pre-existing network or not
	if os.path.exists(PATH) and True:
		net.load_state_dict(torch.load(PATH))
		print("Weights loaded - if this was not desired, please disable this in the code (line 45) and rerun.")

	# loads CNN to device
	net.to(device)

	# gets class names
	paths, dirs, files = os.walk("data/train").__next__()
	classes = dirs

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

	# training loop
	if train:
		for epoch in range(1):  # loop over the dataset multiple times

		    running_loss = 0.0
		    for i, data in enumerate(trainloader, 0):
		        # get the inputs; data is a list of [inputs, labels]
		        inputs, labels = data[0].to(device), data[1].to(device)

		        # zero the parameter gradients
		        optimizer.zero_grad()

		        # forward + backward + optimize
		        outputs = net(inputs)
		        loss = criterion(outputs, labels)
		        loss.backward()
		        optimizer.step()

		        # print statistics
		        running_loss += loss.item()
		        if i % 500 == 499:    # print every 2000 mini-batches
		            print('[%d, %5d] loss: %.3f' %
		                  (epoch + 1, i + 1, running_loss / 500))
		            running_loss = 0.0

		print('Finished Training')

		# save the trained network for use with the hand-gesture HCI system
		torch.save(net.state_dict(), PATH)

	# test the accuracy of the network against unknown test data
	correct = 0
	total = 0
	with torch.no_grad():
	    for data in testloader:
	        images, labels = data[0].to(device), data[1].to(device)
	        outputs = net(images)
	        _, predicted = torch.max(outputs.data, 1)
	        total += labels.size(0)
	        correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the ' + str(total) + ' test images: %d %%' % (
	    100 * correct / total))

	class_correct = list(0. for i in range(10))
	class_total = list(0. for i in range(10))
	with torch.no_grad():
	    for data in testloader:
	        images, labels = data[0].to(device), data[1].to(device)
	        outputs = net(images)
	        _, predicted = torch.max(outputs, 1)
	        c = (predicted == labels).squeeze()
	        for i in range(4):
	            label = labels[i]
	            class_correct[label] += c[i].item()
	            class_total[label] += 1


	for i in range(len(classes)):
	    print('Accuracy of %5s : %2d %%' % (
	        classes[i], 100 * class_correct[i] / class_total[i]))