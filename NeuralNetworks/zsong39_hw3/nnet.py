from Parser import Data_Set
from NeuralNet import NNLearn
import sys

if len(sys.argv) != 6:
	print "Usage: nnet l h e <train-set-file> <test-set-file>"
	exit(1)
[l, h, e, traininput, testinput] = sys.argv[1:]

train_set = Data_Set(traininput)
test_set = Data_Set(testinput)
nn = NNLearn(float(l),int(h),int(e),train_set,test_set)
nn.learn()
nn.classify()
