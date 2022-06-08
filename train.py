from util.main import *
from util.cnn import *
from util.calibrate import *
import argparse

parser = argparse.ArgumentParser(description='Optional camera width and height')
parser.add_argument('-r', type = int, nargs = 2, help = 'Set your resolution by entering the width and height, -r w h')
parser.add_argument('-d', type = int, help = 'Toggle dataset gathering. 0 for off, 1 for on.', default = 1)

args = parser.parse_args()


def main():
	if args.d == 1:
		# get frame size
		if not args.r == None:
			width, height = args.r[0], args.r[1]
			print(width, height)
		else:
			width, height = getCapSize()

		use_cuda = checkCuda()			# check CUDA support on system
		low, high, backSub = calibrate(width, height, use_cuda)			# start calibration process

		getTraining(width, height, low, high, backSub, True, n = 100, test = False)		# get 1000 masks for every gesture, for training
		getTraining(width, height, low, high, backSub, True, n = 20, test = True)			# get 200 masks for every gesture, for testing

	trainCNN()			# begin training process


if __name__ == '__main__':
	main()