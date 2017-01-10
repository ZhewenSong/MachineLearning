from Parser import Data_Set
from classify import classify
import sys
if len(sys.argv) != 4:
	print "Usage: kNN <train-set-file> <test-set-file> k"
	exit()
traininput, testinput, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
train_set = Data_Set(traininput)
test_set = Data_Set(testinput)
kls = list(train_set.attribute)[-1]
attribute = train_set.attribute
mae = 0
accuracy = 0
tot = len(test_set.instances)
print 'k value : %d'%k
for i, test_inst in enumerate(test_set.instances):
	predicted = classify(train_set.instances, test_inst, k, attribute, kls)
	actual = test_inst[kls]
	
	if kls == 'response':
		mae += abs(predicted - actual)
		print 'Predicted value : %.6f	Actual value : %.6f'%(predicted, actual)
	else:
		if predicted == actual:
			accuracy += 1
		print 'Predicted class : %s   Actual class : %s'%(predicted, actual)
if kls == 'response':
	print 'Mean absolute error : %.16f'%(1.*mae/tot)
	print 'Total number of instances : %d'%tot
else:
	print 'Number of correctly classified instances : %d'%accuracy
	print 'Total number of instances : %d'%tot
	print 'Accuracy : %.16f'%(1.*accuracy/tot)