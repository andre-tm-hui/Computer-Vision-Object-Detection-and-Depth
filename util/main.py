import cv2
import numpy as np
import time
from util.calibrate import *
import math
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
import os;
from torch.utils.data import Dataset, DataLoader
from util.gestureControls import *
from PIL import Image

# gesture dictionary, containing all the useful gesture data. This is to be loaded onto a shared
# variable/queue between two processes
gesture = {
	"previous": "",
	"current": "",
	"duration": 0,
	"next": "",
	"frames_elapsed": 0,
	"frame_limit": 4,
	"mouse_position": (0,0),
	"width": 0
}

##################################################################################
# neural network stuff

def gestureCNN(use_cuda):
	device = torch.device("cuda:0" if (torch.cuda.is_available() and use_cuda) else "cpu")
	#print(device) # uncomment to see what device your CNN is using
	# instantiate a Resnet netwrok and load the pre-trained weights
	net = torchvision.models.wide_resnet50_2()
	net.load_state_dict(torch.load("./util/handModel_5g.pth"))
	net.to(device)
	# set to evaluation mode
	net.eval()
	paths, classes, files = os.walk("./data/train").__next__()

	# set the transformation to be applied to the input image
	data_transform = transforms.Compose([
	    transforms.Resize(256),
	    transforms.ToTensor(),
	    transforms.Normalize(mean=[0.485, 0.456, 0.406],
	                         std=[0.229, 0.224, 0.225])
	])

	return device, net, classes, data_transform

def useNet(device, net, classes, data_transform, hand, use_cuda):
	try:
		# convert grayscale mask to rgb mask, since Resnet requires a 3-channel image
		hand = cv2.cvtColor(cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
		# show the mask to the user
		cv2.imshow("hand", hand)

		# convert image from numpy to PIL image
		image = Image.fromarray(hand)
		image = data_transform(image).unsqueeze(0)

		# set the image type, depending on the device used for the CNN
		if use_cuda:
			image = image.type(torch.cuda.FloatTensor)
		else:
			image = image.type(torch.FloatTensor)
		image.to(device)

		# get the confidence values
		with torch.no_grad():
			if use_cuda:
				outputs = net(image.cuda())
			else:
				outputs = net(image)

		# convert confidence values into a set of probabilities
		p = torch.nn.functional.softmax(outputs[0], dim=0)
		class_index = p.argmax(0)
		
		# ensure high confidence in the prediction before updating the gesture data
		if p[class_index].item() > 0.8:
			if gesture["next"] == classes[class_index]:
				gesture["frames_elapsed"] += 1
				# the gesture needs to be detected multiple times before the system is completely confident the gesture has changed
				if gesture["frames_elapsed"] >= gesture["frame_limit"] and gesture["current"] != gesture["next"]:
					gesture["previous"] = gesture["current"]
					gesture["current"] = classes[class_index]
					gesture["frames_elapsed"] = 0
			else:
				# if the next gesture is different to the predicted gesture, set it to the newly predicted gesture
				gesture["next"] = classes[class_index]
				gesture["frames_elapsed"] = 1
		else:
			# if confidence is low, set the gesture to n/a, i.e. unknown or invalid
			if gesture["next"] == "n/a":
				gesture["frames_elapsed"] += 1
				if gesture["frames_elapsed"] >= gesture["frame_limit"] * 3 and gesture["current"] != gesture["next"]:
					gesture["previous"] = gesture["current"]
					gesture["current"] = "n/a"
					gesture["frames_elapsed"] = 0
			else:
				gesture["next"] = "n/a"
				gesture["frames_elapsed"] = 1

		# return the gesture and it's confidence value
		return gesture["current"] + ": " + str(p[class_index].item())
	except:
		# if any errors, return invalid
		return "n/a"

##################################################################################
# test calibration functions

# background subtraction calibration
def calibrateBG(bg_src, use_cuda, resize):
	# set video source
	cap = cv2.VideoCapture(bg_src)
	# initiate background subtractor
	backSub = cv2.createBackgroundSubtractorMOG2(detectShadows = False)

	while True:
		ret, frame = cap.read()
		if ret:
			if resize:
				frame = resizeImage(frame, use_cuda)
			# backSub.apply(src, learningRate), where learningRate defaults to >0, hence it is learning
			bg_mask = backSub.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
			cv2.imshow("background mask", bg_mask)
		else:
			break

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()
	return backSub

# colour thresholding calibration
def calibrateColour(col_src, use_cuda):
	# set video source
	cap = cv2.VideoCapture(col_src)
	low = [0,0,0]
	high = [255,255,255]

	# create sliders for manual adjustment of the low/high colour values
	cv2.namedWindow('calibrate')
	cv2.createTrackbar('lowY','calibrate',low[0],255,none)
	cv2.createTrackbar('highY','calibrate',high[0],255,none)

	cv2.createTrackbar('lowCr','calibrate',low[1],255,none)
	cv2.createTrackbar('highCr','calibrate',high[1],255,none)

	cv2.createTrackbar('lowCb','calibrate',low[2],255,none)
	cv2.createTrackbar('highCb','calibrate',high[2],255,none)

	while True:
		ret, frame = cap.read()
		if ret:
			low = [cv2.getTrackbarPos('lowY', 'calibrate'), cv2.getTrackbarPos('lowCr', 'calibrate'), cv2.getTrackbarPos('lowCb', 'calibrate')]
			high = [cv2.getTrackbarPos('highY', 'calibrate'), cv2.getTrackbarPos('highCr', 'calibrate'), cv2.getTrackbarPos('highCb', 'calibrate')]

			frame = bilateral(frame, use_cuda)

			mask = colourMask(frame, np.array(low), np.array(high))
			cv2.imshow("calibrate", cv2.resize(mask, (1280, 720)))
			cv2.imshow("frame", frame)
		else:
			# loop the same video until colour threshold is properly calibrated
			cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()
	return np.array(low), np.array(high)

def none(e):
	pass

########################################################################
# miscelaneous utility functions

# find the scalar of vector (p1, p2)
def distance(p1, p2):
	return ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) ** 0.5

# resize an image
def resizeImage(frame, use_cuda):
	width = 400			# default width to be resized to - image aspect-ratio will be kept
	height = int(frame.shape[0] * (width / frame.shape[1]))
	if use_cuda:
		c_frame = cv2.cuda_GpuMat()
		c_frame.upload(frame)
		c_frame = cv2.cuda.resize(c_frame, (width, height))
		frame = c_frame.download()
	else:
		frame = cv2.resize(frame, (width, height))
	return frame

# apply a bilateral filter to a frame
def bilateral(frame, use_cuda):
	if use_cuda:
		c_frame = cv2.cuda_GpuMat()
		c_frame.upload(frame)
		c_frame = cv2.cuda.bilateralFilter(c_frame, 10, 25, 25)
		frame = c_frame.download()
	else:
		frame = cv2.bilateralFilter(frame, 10, 25, 25)
	
	return frame

# apply colour thresholding to generate a mask
def colourMask(frame, low, high):
	mask = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
	#mask = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
	#mask = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(mask, low, high)
	mask = morph(mask)

	return mask

# apply morphological transformations to reduce noise in the mask
def morph(mask, size = 7):
	width, height = mask.shape[:2]
	size = int(0.02 * width)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	
	return mask

# checks for CUDA support, mainly for OpenCV
def checkCuda():
	if torch.cuda.is_available():
		use_cuda = True
		# if the bilateral function fails with use_cuda, we assume that OpenCV is not built with CUDA
		try:
			_ = bilateral(np.random.randint(255, size=(50,50,3), dtype=np.uint8), use_cuda)
		except:
			use_cuda = False
	else:
		use_cuda = False
	return use_cuda

# get the frame size of the video source
def getCapSize():
	cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
	cap.set(cv2.CAP_PROP_FPS, 60)
	cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
	
	_, frame = cap.read()
	width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	gesture["frame_limit"] = cap.get(cv2.CAP_PROP_FPS) / 4
	print(width, height)
	cap.release()
	return int(width), int(height)

###########################################################################
# detection functions

# hand segmentation function
def detectHand(frame, backSub, low, high, use_cuda, hand_size, glove):
	height, width = frame.shape[:2]
	min_cov = 0.005			# set the percentage-wise coverage the hand should take up in the frame

	ret = False

	# generate a colour mask
	mask = colourMask(frame, low, high)
	bg_mask = 0

	# if using hand-colour detection, generate a background mask
	if not glove:
		bg_mask = backSub.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), learningRate = 0)
		# get the number of white pixels in the mask
		whitespace = cv2.countNonZero(bg_mask)
		# dynamic update of the background subtractor. set to True if desired, but performance is inconsistent
		if whitespace > 0.15 * width * height and False:
			bg_mask = backSub.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

		# apply morphological transformations
		bg_mask = morph(bg_mask)

		# combine the two masks
		if use_cuda:
			mask = cv2.cuda.multiply(cv2.UMat(mask), cv2.UMat(bg_mask // 255), 1).get()
		else:
			mask = np.multiply(mask, bg_mask // 255)
	
	# find contours in the mask
	contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	if len(contours) > 0:
		# assumption: the contour with the largest area is the contour corresponding to the hand
		hand_contour = max(contours, key = cv2.contourArea)
		# check coverage of hand to ensure it's large enough
		if cv2.contourArea(hand_contour) > (width * height * min_cov):
			# use the minimum bounding rectangle to draw a padded bounding square around the mask
			x,y,w,h = cv2.boundingRect(hand_contour)
			x_m, y_m = int(x + w/2), int(y + h/2)
			box_width = int(1.3 * max(w,h))
			x,y = x_m - box_width // 2, y_m - box_width // 2

			mask_cropped = mask[max(0,y):min(height,y+box_width), max(0,x):min(width,x+box_width)]
			ret = True
		else:
			mask_cropped, x, y, box_width = 0, 0, 0, 0
	else:
		mask_cropped, x, y, box_width = 0, 0, 0, 0

	if ret:
		hand_size = cv2.countNonZero(mask_cropped)
	else:
		hand_size = 0

	return ret, mask_cropped, x, y, box_width, bg_mask, mask, hand_size

# find the fingertip from the mask
def fingertipDefect(mask, x, y, frame_width):
	# make a copy of the image so we're not changing the original
	img = np.copy(mask)
	height, width = mask.shape[:2]

	# find contour of the hand
	contours, _ = cv2.findContours(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	contour = max(contours, key=cv2.contourArea)

	# find convexity defects of the hand contour
	hull = cv2.convexHull(contour, returnPoints = False)
	defects = cv2.convexityDefects(contour, hull)
	clustered = True
	new_defects = []

	for i in range(defects.shape[0]):
		add_defect = True
		s,e,f,d = defects[i][0]
		for j in range(len(hull)):
			if distance(tuple(contour[hull[j][0]][0]), tuple(contour[f][0])) < 0.00003 * width * height:
				add_defect = False
		if add_defect:
			new_defects += [defects[i]]

	if gesture["current"] == "point":
		fingertip0 = [width, height]
		fingertip1 = [width, height]
		for i in range(len(new_defects)):
			s,e,f,d = new_defects[i][0]
			start = tuple(contour[s][0])
			end = tuple(contour[e][0])
			far = tuple(contour[f][0])
			# draw the defects
			cv2.line(img,start,end,[0,255,0],2)
			cv2.circle(img,far,5,[0,0,255],-1)

			# check adjacent defects
			if start[1] < fingertip0[1]:
				fingertip0 = np.array(start)
			if end[1] < fingertip1[1]:
				fingertip1 = np.array(end)
	else:
		fingertip0 = [0, 0]
		fingertip1 = [width, height]
		for c in contour:
			if c[0][0] > fingertip0[0]:
				fingertip0 = np.array(c[0])
				fingertip1 = np.array(c[0])

	fingertip = np.array(fingertip0 + fingertip1) // 2
	# draw a dot on the fingertip
	cv2.circle(mask, (fingertip[0], fingertip[1]), 5, [255,0,0], -1)
	fingertip = (frame_width - int(max(0,x) + fingertip[0]), int(max(0,y) + fingertip[1]))
	
	# update the gesture data
	if fingertip != (width, height):
		gesture["mouse_position"] = fingertip
		gesture["width"] = distance(fingertip0, fingertip1)

	return img

##########################################################################
# main execution functions

# main hand-gesture detection function
def run(q, main_src, backSub, low, high, use_cuda, resize, width = 0, height = 0, glove = False):
	device, net, classes, data_transform = gestureCNN(use_cuda)

	# set time for later fps calculations
	t = time.time()
	point_count = 0
	# check if the source type is a file or webcam stream
	if type(main_src) == str:
		cap = cv2.VideoCapture(main_src)
		frame_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		if height > width:
			width, height = height, width
		if not os.path.exists(main_src[:-7] + str(width) + str(height) + ".txt"):
			with open(main_src[:-7] + str(width) + str(height) + ".txt", "w+") as f:
				f.write("res")
		if cap.get(cv2.CAP_PROP_FPS) > 35 and not os.path.exists(main_src[:-7] + "60fps.txt"):
			with open(main_src[:-7] + "60fps.txt", "w+") as f:
				f.write("fps")
		if glove and not os.path.exists(main_src[:-7] + "glove.txt"):
			with open(main_src[:-7] + "glove.txt", "w+") as f:
				f.write("glove")
	else:
		cap = cv2.VideoCapture(main_src, cv2.CAP_DSHOW)
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
		cap.set(cv2.CAP_PROP_FPS, 60)
		cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
		frame_n = 0

	# initialise variables
	inp, frame, hand_mask, defects, bg_mask, full_mask, hand_size = 0, 0, 0, 0, 0, 0, 0
	while True:
		# check if gesture controls are activated
		if not q == 0:
			q.put(gesture)
		ret, frame = cap.read()

		if ret:
			gestureText = np.zeros((30, 900, 3), np.uint8)
			if resize:
				frame = resizeImage(frame, use_cuda)
			frame = bilateral(frame, use_cuda)
			inp = np.copy(frame)
			# get hand mask
			found, hand_mask, x, y, box_width, bg_mask, full_mask, hand_size = detectHand(frame, backSub, low, high, use_cuda, hand_size, glove)
			if found:
				hand_mask = cv2.cvtColor(hand_mask, cv2.COLOR_GRAY2BGR)
				# get the gesture prediction
				captionText = useNet(device, net, classes, data_transform, hand_mask, use_cuda)
				if captionText == "n/a":
					hand_size = 0
				if gesture['current'] == 'point' or gesture['previous'] == 'point':
					defects = fingertipDefect(hand_mask, x, y, width)
				# for result-gathering purposes
				if gesture['current'] == 'point':
					point_count += 1

				# write gesture data to gestureText window
				cv2.putText(gestureText, captionText, (3, 27), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

				# draw a bounding box with gesture info around the hand
				padding = int(0.02 * box_width)
				cv2.rectangle(frame, (x-padding,y-padding), (x+box_width+padding, y+box_width+padding), (0,0,255), 2)
				cv2.putText(frame, captionText, (x-padding, y-padding-12), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

			# show images to the user
			cv2.imshow("frame", frame)
			cv2.imshow("hand", hand_mask)
			cv2.imshow("gesture", gestureText)
		else:
			break

		if cv2.waitKey(1) & 0xFF == ord('q'):
			# terminate the process by pressing "q"
			break
		elif cv2.waitKey(1) == ord('r'):
			# save images for a particular frame by pressing "r"
			try:
				if not os.path.exists('outputImages'):
					os.mkdir('outputImages')
				print('saving')
				cv2.imwrite('outputImages/input.png', inp)
				cv2.imwrite('outputImages/output.png', frame)
				cv2.imwrite('outputImages/cropped_mask.png', hand_mask)
				cv2.imwrite('outputImages/defects.png', defects)
				cv2.imwrite('outputImages/bg_mask.png', bg_mask)
				cv2.imwrite('outputImages/full_mask.png', full_mask)
			except:
				pass
					

	cap.release()
	cv2.destroyAllWindows()
	return frame_n, (time.time() - t), point_count

# main function to be run when initialising program
def main(q, width, height, subject = -1, experiment = True, glove = False, resize = False, use_cuda = True):
	# get CUDA availability
	if use_cuda:
		use_cuda = checkCuda()

	# check if we are currently trying to get results
	if experiment:
		PATH = 'expfiles/test' + str(subject) + '/'

		if os.path.exists(PATH + 'cfg.npy'):
			low, high = np.load(PATH + 'cfg.npy')
		else:
			col = calibrateColour(PATH + 'col.mp4', use_cuda)
			low, high = col[0], col[1]
			np.save(PATH + 'cfg.npy', np.array([low, high]))

		backSub = calibrateBG(PATH + 'bg.mp4', use_cuda, resize)
		frame_n, t_elapsed, total_point = run(q, PATH + 'main.mp4', backSub, low, high, use_cuda, resize)
		frame_n_point, _, real_point = run(q, PATH + 'point.mp4', backSub, low, high, use_cuda, resize)

		if use_cuda:
			fname = "results_cuda.txt"
		else:
			fname = "results.txt"
		with open(PATH + fname, 'w+') as f:
			f.write(str(frame_n) + ', ' + str(t_elapsed))
		with open(PATH + 'point.txt', 'w+') as f:
			f.write(str(frame_n_point) + ', ' + str(real_point) + ', ' + str(total_point))

	else:
		# run the interactive calibration/program otherwise
		low, high, backSub = calibrate(width, height, use_cuda, glove)
		_, _, _ = run(q, 0, backSub, low, high, use_cuda, False, width, height, glove)
	return low, high, backSub

# function for obtaining training data
def getTraining(width, height, low, high, backSub, use_cuda, n = 1000, test = False):
	# list of gesture names
	gestureList = ['peaceup', 'peacedown', 'pinch', 'point', 'ok']

	cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
	cap.set(cv2.CAP_PROP_FPS, 60)
	cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

	for gesture in gestureList:
		print(gesture)
		if test:
			path = 'data/test/' + gesture + '/'
		else:
			path = 'data/train/' + gesture + '/'
		if not os.path.exists(path):
			os.mkdir(path)

		ready = False
		i, hand_size = 0, 0
		while True:
			_, frame = cap.read()

			frame = bilateral(frame, use_cuda)
			found, hand_mask, x, y, box_width, bg_mask, full_mask, hand_size = detectHand(frame, backSub, low, high, use_cuda, hand_size, glove = True)
			if found:
				hand_mask = cv2.cvtColor(hand_mask, cv2.COLOR_GRAY2BGR)
				if ready:
					cv2.imwrite(path + str(i) + '.png', hand_mask)
					i += 1
				if i >= n:
					# break the loop once enough masks are gathered
					break

			cv2.imshow("frame", frame)
			cv2.imshow("hand", hand_mask)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				# start recording on downpress of "q"
				ready = True

	cap.release()
	cv2.destroyAllWindows()
	return



	