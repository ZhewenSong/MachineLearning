from Parser import Data_Set
from BayesNet import BNLearn as BN
import sys

if len(sys.argv) != 4:
	print "Usage: bayes <train-set-file> <test-set-file> <n|t>"
	exit(1)
[traininput, testinput, option] = sys.argv[1:]

train = Data_Set(traininput)
test = Data_Set(testinput)
train_set = train.instances
test_set = test.instances
attribute = train.attribute
label = train.label
bn = BN(train_set,test_set,attribute,label,option)
bn.classify()

