import numpy as np
import cv2
import yolo as y
import math, os, time, argparse

# parse arguments

parser = argparse.ArgumentParser(description='Run program')

parser.add_argument("-hg", "--histogram", type=int, help="toggle between histogram mean and overall mode", default=1)
parser.add_argument("-o", "--overlap", type=int, help="enable overlap removal", default=1)
parser.add_argument("-c", "--confidence", type=int, help="show/hide confidence values", default=0)

args = parser.parse_args()

# point to data directory

master_path_to_dataset = "TTBB-durham-02-10-17-sub10"; # ** need to edit this **
directory_to_cycle_left = "left-images";     # edit this if needed
directory_to_cycle_right = "right-images";   # edit this if needed

crop_disparity = False; # display full or cropped disparity image
pause_playback = False; # pause until key press after each image

#####################################################################

# resolve full directory location of data set for left / right images

full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left);
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right);

# get a list of the left image files and sort them (by timestamp in filename)

left_file_list = sorted(os.listdir(full_path_directory_left));

##########################################################################

camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres

#########################################################

# gamma correction

def adj_gamma(img, gamma=1.0):
    invGamma = 1.0/gamma
    table = np.array([((i/255.0)**invGamma)*255 for i in np.arange(0,256)]).astype("uint8")
    return cv2.LUT(img, table)

#########################################################
# ORB feature matching

# Initiate detector and FLANN params
try:
	# in newer versions of opencv2, SURF and SIFT are disabled
	feature_object = cv2.xfeatures2d.SURF_create(400)
	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 1)
	print(True)
except:
	feature_object = cv2.ORB_create(800)
	FLANN_INDEX_LSH = 6
	index_params= dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1) #2
search_params = dict(checks=50)

def getDistances(img1, img2):
	distances = np.zeros(img1.get().shape)
	f = camera_focal_length_px
	B = stereo_camera_baseline_m
	# find the keypoints and descriptors with SIFT
	kp1, des1 = feature_object.detectAndCompute(img1,None)
	kp2, des2 = feature_object.detectAndCompute(img2,None)

	flann = cv2.FlannBasedMatcher(index_params,search_params)

	matches = flann.knnMatch(des1,des2,k=2)
	#print(matches)

	# Need to draw only good matches, so create a mask
	goodMatches = []
	matchesMask = [[0,0] for i in range(len(matches))]

	# ratio test as per Lowe's paper
	try:
		for i,mat in enumerate(matches):
		    if len(mat) == 2 and mat[0].distance < 0.7*mat[1].distance:
		        goodMatches += [mat]
		        matchesMask[i] = [1,0]
	except:
		print(matches)

	for mat in goodMatches:
		img1_idx = mat[0].queryIdx
		img2_idx = mat[0].trainIdx

		(x1,y1) = kp1[img1_idx].pt
		(x2,y2) = kp2[img2_idx].pt

		pyth = math.sqrt((x1-x2)**2 + (y1-y2)**2)

		if pyth != 0:
			distances[int(y1)][int(x1)] = f * B / pyth

	draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask = matchesMask, flags = 0)

	img3 = cv2.drawMatchesKnn(img1.get(),kp1,img2.get(),kp2,matches,None,**draw_params)

	cv2.imshow("matches", img3)

	return distances

#########################################################

# running code

total_t = 0
file_counter = 0
for filename_left in left_file_list:
    # from the left image filename get the correspondoning right image

    filename_right = filename_left.replace("_L", "_R");
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left)
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right)

    # check the file is a PNG file (left) and check a correspondoning right image
    # actually exists

    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :

        # read left and right images and display in windows
        # N.B. despite one being grayscale both are in fact stored as 3-channel
        # RGB images so load both as such

        # start a timer
        start_t = time.time()

        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

        # make a separate copy of the image to pass to yolo object detection
        outL = np.copy(imgL)
        
        print("-- files loaded successfully");
        print();

        # gamma correction and power law transformation to improve disparity maps
        imgL = adj_gamma(imgL, 1.5)
        imgL = np.power(imgL, 0.8).astype("uint8")
        imgR = adj_gamma(imgR, 1.5)
        imgR = np.power(imgR, 0.8).astype("uint8")
        imgL = cv2.UMat(imgL)
        imgR = cv2.UMat(imgR)
        
        # remember to convert to grayscale (as the disparity matching works on grayscale)
        # N.B. need to do for both as both are 3-channel images

        grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)

        # image denoising and sharpening using unsharp masking
        temp = cv2.GaussianBlur(grayL,(11,11),10)
        grayL = cv2.addWeighted(grayL,1.5,temp,-0.5,0)
        temp = cv2.GaussianBlur(grayR,(11,11),10)
        grayR = cv2.addWeighted(grayR,1.5,temp,-0.5,0)

        disp = getDistances(grayL,grayR)
        output_image = y.drawBoxes(outL, disp, args.histogram == 1, args.overlap == 1, args.confidence == 1, True)

        # stop the timer, and draw it to the output image
        t = time.time() - start_t
        label = 'Inference time: %.2f s' % (t)
        cv2.putText(output_image, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        
        # keep track of total time and image index for average time calculation
        total_t += t 
        file_counter += 1

        # show the images
        cv2.imshow("image", output_image)

        cv2.imwrite("sparse-output/" + filename_left, output_image)
        
        # keyboard input for exit (as standard), save disparity and cropping
        # exit - x
        # save - s
        # crop - c
        # pause - space

        key = cv2.waitKey(40 * (not(pause_playback))) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        if (key == ord('x')):       # exit
            break; # exit
        elif (key == ord('s')):     # save
            cv2.imwrite("sgbm-disparty-" + filename_left, disparity_scaled);
            cv2.imwrite(filename_left, imgL);
            cv2.imwrite(filename_right, imgR);
            cv2.imwrite("out-" + filename_left, output_image)
        elif (key == ord('c')):     # crop
            crop_disparity = not(crop_disparity);
        elif (key == ord(' ')):     # pause (on next frame)
            pause_playback = not(pause_playback);
    else:
            print("-- files skipped (perhaps one is missing or not PNG)");
            print();

    # write average time per image to a text file after each loop
    with open("avg.txt","w") as f:
        f.write(str(total_t/file_counter))

# close all windows
cv2.destroyAllWindows()

#####################################################################