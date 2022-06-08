from util.main import *
from util.gestureControls import *
import multiprocessing
import argparse

parser = argparse.ArgumentParser(description='Optional camera width and height')
parser.add_argument('-r', type = int, nargs = 2, help = 'Set your resolution by entering the width and height, -r w h')

args = parser.parse_args()

glove = True

if __name__ == '__main__':
	# get frame size
	if not args.r == None:
		width, height = args.r[0], args.r[1]
		print(width, height)
	else:
		width, height = getCapSize()

	# create a variable that can be shared between processes
	q = multiprocessing.Queue()
	# configure two processes, with the shared queue
	p1 = multiprocessing.Process(target = main, args = (q, width, height, -1, False, glove))
	p2 = multiprocessing.Process(target = gestureToFunction, args = (q, width, height,))
	# start the processes
	p1.start()
	p2.start()