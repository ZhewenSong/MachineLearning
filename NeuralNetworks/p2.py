from Parser import Data_Set
from NeuralNet import NNLearn
from matplotlib import pyplot as plt

traininput = 'heart_train.arff'
testinput = 'heart_test.arff'

l = 0.1
train_set = Data_Set(traininput)
test_set = Data_Set(testinput)
epoch = [1, 10, 100, 500]
for h in [0, 20]:
	plt.figure()
	train = []
	test = []
	for e in epoch:
		nn = NNLearn(l, h, e, train_set, test_set)
		train.append(1 - nn.learn())
		test.append(1 - nn.classify()[0])
	plt.plot(epoch, train, 'o', epoch, test, 'x')
	plt.legend(['train', 'test'])
	plt.title('%d hidden units'%h)
	plt.xlabel('# epochs')
	plt.ylabel('error rate')
	plt.xlim([0, 600])
	plt.ylim([0, 0.5])
	plt.show()