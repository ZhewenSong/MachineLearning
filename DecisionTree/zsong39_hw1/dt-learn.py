from Parser import Data_Set
from Dectree import Dectree_learn
import sys

if len(sys.argv) != 4:
	print "Usage: dt-learn <train-set-file> <test-set-file> m"
	exit(1)
traininput, testinput, m = sys.argv[1], sys.argv[2], int(sys.argv[3])
train_set = Data_Set(traininput)
test_set = Data_Set(testinput)
attributes = train_set.attribute
cls_values = train_set.label
dectree = Dectree_learn(train_set, test_set, m)
tmp_label = dectree.get_default_label(train_set.instances)
if tmp_label == None:
	tmp_label = cls_values[0]
root = dectree.build_tree(train_set.instances, attributes.copy(), tmp_label)
dectree.print_tree(root, 0)
print '<Predictions for the Test Set Instances>\r'
cnt = 0
for i, inst in enumerate(test_set.instances):
	predicted = dectree.classify(root, inst)
	if predicted == inst['class']:
		cnt += 1
	print '%d: Actual: %s Predicted: %s\r'%(i+1, inst['class'], predicted)
print 'Number of correctly classified: %d Total number of test instances: %d\r'%(cnt, len(test_set.instances))

