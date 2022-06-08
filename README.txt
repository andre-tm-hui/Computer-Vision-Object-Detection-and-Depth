DISCLAIMER: This program has only been tested on Windows 10 machines. Mileage may vary on Linux/Macintosh OS.

Prerequisites:
- Python 3.7 (64 bit) with the following libraries:
	- OpenCV (for CUDA support, see here: https://jamesbowley.co.uk/)
	- Pytorch (refer to: https://pytorch.org)
	- pynput (pip install pynput)
	- wx (pip install wxPython)

File List:
> util
	> calibrate.py
	> cnn.py
	> gestureControls.py
	> main.py
	> handModel_5g.pth
> perftest.html
> run.py
> runGlove.py
> runTests.py
> train.py
> README.txt
> gestures.png

USAGE INSTRUCTIONS:
To use the mouse program:
1. In Command Prompt, execute the python file:
	a. "run.py" for skin-colour based hand segmentation.
	b. "runGlove.py" for glove-based hand segmentation.
   Optional: you can add the following argument "-r *width* *height*" to manually set the capture resolution.
2. There will now be a window corresponding to your webcam feed, titled "frame". This is the calibration phase. When prompted, press "Q" 
   and stay still until the number in the Command Prompt window reaches 200.
3. Place your hand over the green box on the "frame" window, with your open palm facing the webcam, ensuring that your hand covers the 
   entirety of the box, and press "Q".
4. Repeat, whilst making a fist, with your palm still faced towards the webcam.
5. Repeat once more, turning your fist around such that that back of your hand faces the webcam.
6. The calibration process is over - all the windows will close, and a new set of windows will open. Once these windows have opened, you
   may now use the program freely.
7. To stop the program, press "Q" on the "frame" window.

Recognised Gesture Controls:
1. Cursor Movement:
	Make a pointing gesture towards the camera, and move your hand to move the cursor.
2. Left-click:
	Make a pinching gesture towards the camera, and release the gesture to initiate a left-click. If going from the pointing gesture
	to the pinching gesture, you will be allowed to make fine adjustments to the position of the cursor by moving your hand.
3. Right-click:
	Make an "OK" gesture towards the camera, and release the gesture to initiate a right-click. If going from the pointing gesture
	to the "OK" gesture, you will be allowed to make fine adjustments to the position of the cursor by moving your hand.
4. Scrolling:
	Make a 2-fingered gesture, similar to the peace-sign. Flick from up to down to scroll upwards, and down to up to scroll downwards.
Please refer to "gestures.png" for more detailed images of the gestures.

To train the CNN:
1. In Command Prompt, execute the "train.py" python file.
   Optional: you can add the following argument "-r *width* *height*" to manually set the capture resolution.
2. Do the calibration as you would when running the main mouse program (refer to above).
3. Once the new set of windows open, you will see a gesture printed in your Command Prompt window. Hold that gesture in front of the camera,
   and press "Q".
4. Upon pressing "Q", begin to move your hand around the camera frame, maintaining the gesture, until a new gesture is printed in your
   Command Prompt window.
5. Repeat until all gestures (there will be 5) are done. This concludes the gathering of the training data.
6. Repeat again for all 5 gestures, to gather the testing data.
7. The program will now train the CNN. This may takes several hours, depending on your Computer specs.
8. To know when the training is complete, the program will output accuracy results of the trained CNN. The program is preset to do 25 epochs.

To get accuracy and performance results on test data:
1. In the root directory, create a folder "expfiles".
2. For every user you wish to test, create a folder "test*n*", with "n" being an integer corresponding to each user.
3. Edit the "runTests.py" python file such that the "n" variable on line 5 corresponds to the largest "n" in the "expfiles" directory
4. For each user, 4 .mp4 files are required:
	a. "bg.mp4", a short video of the user staying still in front of the camera
	b. "col.mp4", a short video of the user holding their hand in the frame of the camera, rotating it periodically
	c. "main.mp4", a longer video of the user making gestures and moving their hand across the camera frame
	d. "point.mp4", a cropped version of "main.mp4", where only the pointing gesture is made
5. Execute the "runTests.py" python file.
6. If you are running the tests for the first time, for each user, you will be asked to manually set the colour thresholds by adjusting 
   6 sliders. Press "Q" once you are satisfied with the colour thresholding values. This will not be necessary on subsequent runs.
7. Allow the tests to run. Repeat step 6 as required. Results are saved to the "expfiles" directory.
