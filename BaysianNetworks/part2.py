from Parser import Data_Set
from BayesNet import BNLearn as BN
import sys, random

if len(sys.argv) != 2:
	print "Usage: bayes <train-set-file>"
	exit(1)
traininput = sys.argv[1]

train = Data_Set(traininput)
train_set = train.instances  # array
attribute = train.attribute
label = train.label
random.shuffle(train_set)

k = 10
N = len(train_set) 
fold = [None for i in range(k)]
accuracy = {'n':[], 't':[]}
for option in ['n', 't']:
	for i in range(k):
		bn = BN(train_set[:i*(N/k)] + train_set[(i+1)*N/k:],train_set[i*(N/k):(i+1)*N/k],attribute,label,option)
		accu = 1.0 * bn.classify() / (N/k)
		print "Accuracy of fold %d for option %s: %f"%(i, option, accu)
		accuracy[option].append(accu)

print accuracy
from scipy import stats
print stats.ttest_ind(accuracy['t'], accuracy['n'])