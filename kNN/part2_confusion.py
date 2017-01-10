from Parser import Data_Set
from classify import classify
import sys, numpy
from matplotlib import pyplot as plt

traininput = 'yeast_train.arff'
testinput = 'yeast_test.arff'
ks = [1, 30]

train_set = Data_Set(traininput)
test_set = Data_Set(testinput)
attributes = train_set.attribute
classes = attributes['class']
kls = list(train_set.attribute)[-1]
tot = len(test_set.instances)
size = len(classes)

for k in range(len(ks)):
	confusion = numpy.zeros([size, size])
	for i, test_inst in enumerate(test_set.instances):
		predicted = classify(train_set.instances, test_inst, ks[k], attributes, kls)
		actual = test_inst[kls]
		confusion[classes.index(predicted), classes.index(actual)] += 1
	out = 'Predicted \ Actual'
	for c in classes: 
		out += ' %s'%c
	out += '\n'
	for i in range(len(classes)):
		out += '%s              '%classes[i]
		for j in range(len(classes)):
			out += '&%3d'%confusion[i,j]
		out += '\\\\ \n' 
	print(out)

	for i in range(size):
		class_tot = sum(confusion[:, i])
		if class_tot != 0:
			confusion[:, i] /= class_tot
	fig = plt.matshow(confusion, cmap=plt.cm.jet)
	plt.colorbar(fig)
	plt.xticks(range(size), classes)
	plt.xticks(rotation=70)
	plt.yticks(range(size), classes)
	plt.xlabel('Actual Class')
	plt.ylabel('Predicted Class')

	plt.title('$k=%d$'%ks[k], y=1.1)
	plt.show()
