from Parser import Data_Set
from classify import classify
import sys
if len(sys.argv) != 6:
	print "Usage: kNN-select <train-set-file> <test-set-file> k1 k2 k3"
	exit()
traininput, testinput = sys.argv[1], sys.argv[2]
k = [int(sys.argv[i]) for i in range(3, 6)]
train_set = Data_Set(traininput)
test_set = Data_Set(testinput)
kls = list(train_set.attribute)[-1]
attribute = train_set.attribute
mae = [0 for i in range(3)]
misc = [0 for i in range(3)]
train_tot = len(train_set.instances)
for kk in range(3):
	for i, train_inst in enumerate(train_set.instances): 
		predicted = classify([train_set.instances[x] for x in range(train_tot) 
			if x != i], train_inst, k[kk], attribute, kls)
		actual = train_inst[kls]
		if kls == 'response':
			mae[kk] += abs(predicted - actual)
                else:
			if predicted != actual:
				misc[kk] += 1

	if kls == 'response':
		print 'Mean absolute error for k = %d : %.16f'%(k[kk], 1.*mae[kk]/(train_tot))
	else:
		print 'Number of incorrectly classified instances for k = %d : %d'%(k[kk], misc[kk])
if kls == 'response':
	kbest = k[mae.index(min(mae))]
else:
	kbest = k[misc.index(min(misc))]
print 'Best k value : %d'%kbest

test_tot = len(test_set.instances)
mae = 0
accuracy = 0
for i, test_inst in enumerate(test_set.instances):
	predicted = classify(train_set.instances, test_inst, kbest, attribute, kls)
	actual = test_inst[kls]
	
	if kls == 'response':
		mae += abs(predicted - actual)
		print 'Predicted value : %.6f	Actual value : %.6f'%(predicted, actual)
	else:
		if predicted == actual:
			accuracy += 1
		print 'Predicted class : %s   Actual class : %s'%(predicted, actual)
if kls == 'response':
	print 'Mean absolute error : %.16f'%(1.*mae/test_tot)
	print 'Total number of instances : %d'%test_tot
else:
	print 'Number of correctly classified instances : %d'%accuracy
	print 'Total number of instances : %d'%test_tot
	print 'Accuracy : %.16f'%(1.*accuracy/test_tot)
