from Dectree import Dectree_learn
from Parser import Data_Set
import random, sys, numpy
from matplotlib import pyplot as plt

ds, m = sys.argv[1], 4
traininput = ds+'_train.arff'
testinput = ds+'_test.arff'

train_set = Data_Set(traininput)
test_set = Data_Set(testinput)
attributes = train_set.attribute
cls_values = train_set.label
dectree = Dectree_learn(train_set, test_set, m)
size = numpy.array([0.05, 0.1, 0.2, 0.5, 1.0])
cnt = numpy.array([[0 for i in range(10)] for j in range(len(size))])

for j,s in enumerate(size):
    for i in range(10):
        sample_instances = random.sample(train_set.instances, int(s*len(train_set.instances)))
    	tmp_label = dectree.get_default_label(sample_instances)
        if tmp_label == None:
            tmp_label = cls_values[0]
        root = dectree.build_tree(sample_instances, attributes.copy(), tmp_label)
    	for inst in test_set.instances:
    	    predicted = dectree.classify(root, inst)
    	    if predicted == inst['class']:
    		cnt[j][i] += 1

plt.plot(size*100, [1.0*max(cnt[j,:])/len(test_set.instances)*100 for j in range(len(size))], 'o', 
    size*100, [1.0*min(cnt[j,:])/len(test_set.instances)*100 for j in range(len(size))], 's', 
    size*100, [1.0*sum(cnt[j,:])/10/len(test_set.instances)*100 for j in range(len(size))], 'x')
plt.xlabel('training set %')
plt.ylabel('accuracy %')
plt.title('Learning curve (%s)'%ds)
plt.legend(['max','min','ave'])
plt.ylim([50,90])
plt.show()

