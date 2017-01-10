from Parser import Data_Set
from NeuralNet import NNLearn
from matplotlib import pyplot as plt
import numpy as np

for case in ['heart', 'lymph']:
	traininput = '%s_train.arff'%case
	testinput = '%s_test.arff'%case
	train_set = Data_Set(traininput)
	test_set = Data_Set(testinput)
	nn = NNLearn(0.1, 20, 100, train_set, test_set)
	nn.learn()
	data = nn.classify()[1]
	threshold = np.arange(0.01, 1.0, 0.01)
	ROC = []
	plt.figure()
	for th in threshold:
		TP, FP, TN, FN = 0, 0, 0, 0
		for i in range(len(data)):
			if data[i, 1] > th:
				if data[i, 0] == 1:
					TP += 1
				else:
					FP += 1
			else:
				if data[i, 0] == 0:
					TN += 1
				else:
					FN += 1
		ROC.append([1.*FP/(TN+FP), 1.*TP/(TP+FN)])
	ROC = np.array(ROC)
	plt.plot(ROC[:,0], ROC[:,1], 'o')
	plt.xlim([0,1])
	plt.ylim([0,1])
	plt.title('ROC of %s data'%case)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.show()

