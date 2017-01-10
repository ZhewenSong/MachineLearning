from Dectree import Dectree_learn
from Parser import Data_Set
import random, sys
from matplotlib import pyplot as plt

traininput, testinput = sys.argv[1], sys.argv[2]

train_set = Data_Set(traininput)
test_set = Data_Set(testinput)
attributes = train_set.attribute
cls_values = train_set.label
m = [2, 5, 10, 20]
accuracy = []
for mm in m:
    cnt = 0
    dectree = Dectree_learn(train_set, test_set, mm)
    sample_instances = train_set.instances
    tmp_label = dectree.get_default_label(sample_instances)
    if tmp_label == None:
        tmp_label = cls_values[0]
    root = dectree.build_tree(sample_instances, attributes.copy(), tmp_label)
    for i, inst in enumerate(test_set.instances):
	    predicted = dectree.classify(root, inst)
	    if predicted == inst['class']:
		cnt += 1
    accuracy.append(100.0*cnt/(len(test_set.instances)))

plt.plot(m, accuracy, 'o')
plt.xlabel('m')
plt.ylabel('accuracy %')
plt.title('Learning curve (diabetes)')
plt.ylim([50,75])
plt.show()

