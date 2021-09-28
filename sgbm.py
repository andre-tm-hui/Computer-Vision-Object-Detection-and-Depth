#####################################################################

# Example : load, display, detect objects and compute SGBM disparity
# for a set of rectified stereo images from a directory structure
# of left-images / right-images with filesname DATE_TIME_STAMP_{L|R}.png

# Author : Andre Hui

# This code: significant portions of the code is taken from an example
# by T. Breckon, available at:
# https://github.com/tobybreckon/stereo-disparity

#####################################################################

import cv2
import os
import numpy as np
import yolo as y
import time
import argparse

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

# setup the disparity stereo processor to find a maximum of 128 disparity values

max_disparity = 128;
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);

#########################################################

# gamma correction

def adj_gamma(img, gamma=1.0):
    invGamma = 1.0/gamma
    table = np.array([((i/255.0)**invGamma)*255 for i in np.arange(0,256)]).astype("uint8")
    return cv2.LUT(img, table)

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
        imgL = adj_gamma(imgL, 0.75)
        imgL = np.power(imgL, 0.8).astype("uint8")
        imgR = adj_gamma(imgR, 0.75)
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

        # compute disparity image from undistorted and rectified stereo images
        # that we have loaded
        # (which for reasons best known to the OpenCV developers is returned scaled by 16)

        disparity = stereoProcessor.compute(grayL.get(),grayR.get())

        # filter out noise and speckles (adjust parameters as needed)

        dispNoiseFilter = 5; # increase for more agressive filtering
        cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter)

        # scale the disparity to 8-bit for viewing
        # divide by 16 and convert to 8-bit image (then range of values should
        # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
        # so we fix this also using a initial threshold between 0 and max_disparity
        # as disparity=-1 means no disparity available

        _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO)
        disparity_scaled = (disparity / 16.).astype(np.uint8)

        # crop disparity to chop out left part where there are with no disparity
        # as this area is not seen by both cameras and also
        # chop out the bottom area (where we see the front of car bonnet)

        if crop_disparity:
            width = np.size(disparity_scaled, 1)
            disparity_scaled = disparity_scaled[0:390,135:width]

        # display image (scaling it to the full 0->255 range based on the number
        # of disparities in use for the stereo part)

        output_disparity = (disparity_scaled * (256. / max_disparity)).astype(np.uint8)
        backtorgb = cv2.cvtColor(output_disparity,cv2.COLOR_GRAY2RGB)
        output_image = y.drawBoxes(outL, output_disparity, args.histogram == 1, args.overlap == 1, args.confidence == 1, False)

        # stop the timer, and draw it to the output image
        t = time.time() - start_t
        label = 'Inference time: %.2f s' % (t)
        cv2.putText(output_image, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        
        # keep track of total time and image index for average time calculation
        total_t += t 
        file_counter += 1

        # show the images
        cv2.imshow("image", output_image)
        cv2.imshow("disparity", output_disparity)

        cv2.imwrite("sgbm-output/" + filename_left, output_image)
        
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

