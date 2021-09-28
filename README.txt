Extract the source folder to any directory.

It is recommended that you use an Anaconda environment with python 3.7 and the following modules installed:
	- numpy - conda install numpy
	- opencv - conda install opencv
	- pytorch - conda install pytorch torchvision *cuda-version-here* -c pytorch
Replace *cuda-version-here* with:
	- cudatoolkit = 9.2
	- cudatoolkit = 10.1
	- cpuonly
depending on which version of CUDA is installed.

The software has also been tested on MIRA with pytorch and opencv enabled.

To use this program, ensure you have the following files:
	- yolov3.weights -> https://pjreddie.com/media/files/yolov3.weights
    	- yolov3.cfg -> https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true
    	- coco.names -> https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true
located in the same directory as sgbm.py and sparse.py, in the source folder.

Ensure you have properly indicated where your set of stereo images are located in:
	- sgbm.py at line 34-36
	- sparse.py at line 18-20

Execute sgbm.py or sparse.py with 0 or more options:
	- "-hg _" -> toggles between using a histogram (1) or the mode (0) to find a representative disparity value
	- "-o _" -> enable (1) or disable (0) the overlap removal algorithm
	- "-c _" -> display (1) or hide (0) the confidence values in the output image

Whilst the program is executing, press:
	- "x" to quit
	- "s" to save the current frame to file
	- "c" to toggle the cropping of the disparity map
	- " " to pause execution

