from util.main import *

n = 10

def runTests(q, width, height):
	# test data should be in an "expfiles/test*n*" folder, with each test*n* folder corresponding to a unique set of test data
	for i in range(0, n):
		try:
			try:
				# run tests with CUDA
				low, high, backSub = main(q, width, height, i, True, use_cuda = True)
			except:
				# catch exceptions in the case CUDA is not supported
				pass
			# then run tests without CUDA
			low, high, backSub = main(q, width, height, i, True, use_cuda = False)
		except:
			pass
	calcResults()

############################################################################################
# calcResults() goes through all the saved data from the runTests() function, and automatically
# calculates precision, recall, f1-score and fps

def calcResults():
	rootDir = "expfiles"
	results = {
		"19201080": {
			"cuda": {
				"frames": 0,
				"time": 0
			},
			"nocuda": {
				"frames": 0,
				"time": 0
			}
		},
		"1280720": {
			"cuda": {
				"frames": 0,
				"time": 0
			},
			"nocuda": {
				"frames": 0,
				"time": 0
			}
		},
		"640480": {
			"cuda": {
				"frames": 0,
				"time": 0
			},
			"nocuda": {
				"frames": 0,
				"time": 0
			}
		},
		"30fps": {
			"glove": {
				"tp": 0,
				"fn": 0,
				"fp": 0
			},
			"noglove": {
				"tp": 0,
				"fn": 0,
				"fp": 0
			},
		},
		"60fps": {
			"glove": {
				"tp": 0,
				"fn": 0,
				"fp": 0
			},
			"noglove": {
				"tp": 0,
				"fn": 0,
				"fp": 0
			},
		}

	}
	for test in os.listdir(rootDir):
		testDir = rootDir + "/" + test + "/"
		print(testDir)
		if os.path.exists(testDir + "19201080.txt"):
			res = "19201080"
		elif os.path.exists(testDir + "1280720.txt"):
			res = "1280720"
		elif os.path.exists(testDir + "640480.txt"):
			res = "640480"
		else:
			res = ""

		if os.path.exists(testDir + "60fps.txt"):
			fps = "60fps"
		else:
			fps = "30fps"

		if os.path.exists(testDir + "glove.txt"):
			glove = "glove"
		else:
			glove = "noglove"

		print(res, fps, glove)

		if len(res) > 1:
			try:
				with open(testDir + "point.txt", "r") as f:
					temp = f.readline().replace("\n", "").split(", ")
					if int(temp[0]) > int(temp[1]) and int(temp[2]) > int(temp[1]):
						results[fps][glove]["tp"] += int(temp[1])
						results[fps][glove]["fn"] += int(temp[0]) - int(temp[1])
						results[fps][glove]["fp"] += int(temp[2]) - int(temp[1])
			except:
				pass

			try:
				with open(testDir + "results.txt", "r") as f:
					temp = f.readline().replace("\n", "").split(", ")
					results[res]["nocuda"]["frames"] += int(temp[0])
					results[res]["nocuda"]["time"] += float(temp[1])
			except:
				pass

			try:
				with open(testDir + "results_cuda.txt", "r") as f:
					temp = f.readline().replace("\n", "").split(", ")
					results[res]["cuda"]["frames"] += int(temp[0])
					results[res]["cuda"]["time"] += float(temp[1])
			except:
				pass

		print(results)

	with open(rootDir + "/point30glove.txt", "w+") as f:
		tp, fp, fn = results["30fps"]["glove"]["tp"], results["30fps"]["glove"]["fp"], results["30fps"]["glove"]["fn"]
		p, r = tp / (tp + fp), tp / (tp + fn)
		f.write("p %6f, r %6f, f1 %6f" % (p, r, (2 * p * r) / (p + r)))

	with open(rootDir + "/point30noglove.txt", "w+") as f:
		tp, fp, fn = results["30fps"]["noglove"]["tp"], results["30fps"]["noglove"]["fp"], results["30fps"]["noglove"]["fn"]
		p, r = tp / (tp + fp), tp / (tp + fn)
		f.write("p %6f, r %6f, f1 %6f" % (p, r, (2 * p * r) / (p + r)))

	with open(rootDir + "/point60glove.txt", "w+") as f:
		tp, fp, fn = results["60fps"]["glove"]["tp"], results["60fps"]["glove"]["fp"], results["60fps"]["glove"]["fn"]
		p, r = tp / (tp + fp), tp / (tp + fn)
		f.write("p %6f, r %6f, f1 %6f" % (p, r, (2 * p * r) / (p + r)))

	with open(rootDir + "/point60noglove.txt", "w+") as f:
		tp, fp, fn = results["60fps"]["noglove"]["tp"], results["60fps"]["noglove"]["fp"], results["60fps"]["noglove"]["fn"]
		p, r = tp / (tp + fp), tp / (tp + fn)
		f.write("p %6f, r %6f, f1 %6f" % (p, r, (2 * p * r) / (p + r)))

	with open(rootDir + "/results19201080.txt", "w+") as f:
		f.write(str(results["19201080"]["nocuda"]["frames"] / results["19201080"]["nocuda"]["time"]))

	with open(rootDir + "/results1280720.txt", "w+") as f:
		f.write(str(results["1280720"]["nocuda"]["frames"] / results["1280720"]["nocuda"]["time"]))

	with open(rootDir + "/results640480.txt", "w+") as f:
		f.write(str(results["640480"]["nocuda"]["frames"] / results["640480"]["nocuda"]["time"]))

	with open(rootDir + "/results_cuda19201080.txt", "w+") as f:
		f.write(str(results["19201080"]["cuda"]["frames"] / results["19201080"]["cuda"]["time"]))

	with open(rootDir + "/results_cuda1280720.txt", "w+") as f:
		f.write(str(results["1280720"]["cuda"]["frames"] / results["1280720"]["cuda"]["time"]))

	with open(rootDir + "/results_cuda640480.txt", "w+") as f:
		f.write(str(results["640480"]["cuda"]["frames"] / results["640480"]["cuda"]["time"]))

	return

if __name__ == '__main__':
	# get frame size
	width, height = getCapSize()
	runTests(0, width, height)