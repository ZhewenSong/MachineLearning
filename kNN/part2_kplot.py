from Parser import Data_Set
from classify import classify
import sys
from matplotlib import pyplot as plt

ds = sys.argv[1]
traininput = ds+'_train.arff'
testinput = ds+'_test.arff'
if ds == 'yeast':
	ks = [1, 5, 10, 20, 30]
else:
	ks = [1, 2, 3, 5, 10]
train_set = Data_Set(traininput)
test_set = Data_Set(testinput)
attribute = train_set.attribute
kls = list(train_set.attribute)[-1]
tot = len(test_set.instances)
mae = [0 for i in range(len(ks))]
accuracy = [0 for i in range(len(ks))]
for k in range(len(ks)):
	for i, test_inst in enumerate(test_set.instances):
		predicted = classify(train_set.instances, test_inst, ks[k], attribute, kls)
		actual = test_inst[kls]
		if kls == 'response':
			mae[k] += abs(predicted - actual)
		else:
			if predicted == actual:
				accuracy[k] += 1
	if kls == 'response':
		mae[k] = mae[k]*1./tot
	else:
		accuracy[k] = accuracy[k]*1./tot

print mae
if kls == 'response':
	plt.plot(ks, mae, 'o')
	plt.ylabel('Mean Absolute Error')
else:
	plt.plot(ks, accuracy, 'o')	
	plt.ylabel('Accuracy')

plt.xlabel('k')
plt.title(ds)
plt.show()