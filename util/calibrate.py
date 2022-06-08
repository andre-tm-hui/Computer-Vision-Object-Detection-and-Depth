import cv2
import numpy as np

# main function
def calibrate(width, height, use_cuda, glove = False):
	# set capture parameters
	cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
	cap.set(cv2.CAP_PROP_FPS, 60)
	cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

	# set initial conditions
	background = False
	bg_fin = False
	color_mask = {"current": "palm", "fin": False, "min_col": [255,255,255], "max_col": [0,0,0], "y":[], "cb":[], "cr":[]}
	ok = 0
	# initialize background subtractor
	backSub = cv2.createBackgroundSubtractorMOG2(detectShadows = False)

	while True:
		ret, frame = cap.read()

		# apply a bilateral filter
		if use_cuda:
			c_frame = cv2.cuda_GpuMat()
			c_frame.upload(frame)
			c_frame = cv2.cuda.bilateralFilter(c_frame, 10, 25, 25)
			frame = c_frame.download()
			hand_cropped = c_frame.download()
		else:
			frame = cv2.bilateralFilter(frame, 10, 25, 25)

		unprocessed = frame

		if background:
			# train backSub on 200 frames
			if ok < 200:
				bg_mask = backSub.apply(frame)
				ok += 1
				print(ok)
				cv2.putText(frame, "Please stay still", (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 155, 0), 2)
			else:
				bg_fin = True
				bg_mask = backSub.apply(frame, learningRate=0)

			cv2.imshow("bgmask", bg_mask)
		else:
			cv2.putText(frame, "Press Q and stay still for 5 seconds", (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 155, 0), 2)

		# get hand-colour data
		if not color_mask["fin"] and background and bg_fin:
			box_width = width // 20
			# draw a box on the frame for the user's reference
			cv2.rectangle(frame,
				(width//4 - box_width, height//2 - box_width),
				(width//4 + box_width, height//2 + box_width),
				(0, 255, 0),
				2)
			text = color_mask["current"]
			cv2.putText(frame, text, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 155, 0), 2)

		if color_mask["fin"]:
			bg_mask = backSub.apply(frame, learningRate=0)
			mask = cv2.inRange(frame, np.array(color_mask["min_col"]), np.array(color_mask["max_col"]))
			mask *= bg_mask // 255
			cv2.imshow("mask", mask)

		cv2.imshow("frame", frame)

		if color_mask["fin"] and background:
			break
			
		# "q" starts the process
		if cv2.waitKey(1) & 0xFF == ord('q'):
			if not background:
				background = True
			elif not color_mask["fin"] and bg_fin:
				# on "q" press, extract the colour data from the region of interest (ROI)
				box_width = width // 20
				roi = unprocessed[height//2 - box_width:height//2 + box_width, width//4 - box_width:width//4 + box_width]
				cv2.imshow("roi", roi)
				# conversion into appropriate colour space
				roi = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
				
				# add colour data to a list, separated by channels
				y, cr, cb = cv2.split(roi)
				y, cr, cb = y.flatten(), cr.flatten(), cb.flatten()
				color_mask["y"] = np.append(color_mask["y"], y)
				color_mask["cr"] = np.append(color_mask["cr"], cr)
				color_mask["cb"] = np.append(color_mask["cb"], cb)
				
				# iterate throught each gesture/orientation required for colour calibration on each "q" press
				if color_mask["current"] == "palm":
					color_mask["current"] = "fist"
				elif color_mask["current"] == "fist":
					color_mask["current"] = "back"
				elif color_mask["current"] == "back":
					# final "q" press - sort the channel lists
					color_mask["y"].sort(), color_mask["cr"].sort(), color_mask["cb"].sort()
					# set initial values for min/max colour
					min_col, max_col = [int(color_mask["y"][0]), int(color_mask["cr"][0]), int(color_mask["cb"][0])], [int(color_mask["y"][-1]), int(color_mask["cr"][-1]), int(color_mask["cb"][-1])]
					
					# apply the colour threshold and background mask to the frame
					frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
					mask = cv2.inRange(frame, np.array(min_col), np.array(max_col))
					bg_mask = backSub.apply(frame, learningRate=0)
					mask *= bg_mask // 255
					cv2.imshow("mask", mask)
					i = 1
					# set ratio of white-to-black space according to the setup
					ratio = 0.75 if not glove else 0.9
					# continue reducing the range of colour until the ratio is reached
					while cv2.countNonZero(mask) > ratio * width * height:
						min_col, max_col = [int(color_mask["y"][i]), int(color_mask["cr"][i]), int(color_mask["cb"][i])], [int(color_mask["y"][-i-1]), int(color_mask["cr"][-i-1]), int(color_mask["cb"][-i-1])]
						i += 1
						mask = cv2.inRange(frame, np.array(min_col), np.array(max_col))
						mask *= bg_mask // 255
					color_mask["fin"] = True
					color_mask["min_col"], color_mask["max_col"] = np.array(min_col), np.array(max_col)
					color_mask["min_col"][0] = 10
					color_mask["max_col"][0] = 245
			else:
				break

	cap.release()
	cv2.destroyAllWindows()
	return color_mask["min_col"], color_mask["max_col"], backSub