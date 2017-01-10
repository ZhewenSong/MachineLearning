from Parser import Data_Set
import math
from collections import OrderedDict


class Tree_node:
	def __init__(self, label, attribute, attr_count, parent_attribute_value, terminal):
		self.label = label
		self.attribute = attribute
		self.attr_count = attr_count
		self.parent_attribute_value = parent_attribute_value
		self.terminal = terminal
		if terminal:
			self.children = None
		else:
			self.children = []

class Dectree_learn:
	def __init__(self, train_set, test_set, m):
		self.train_set = train_set
		self.test_set = test_set
		self.cls_values = self.train_set.label
		self.m = m


	def build_tree(self, instances, attributes, default_label): 
		if len(attributes) == 0:
			return Tree_node(default_label, None, None, None, True)

		label = instances[0]['class']

		isSameLabel = True
		for inst in instances:
			if inst['class'] != label:
				isSameLabel = False
				break

		if isSameLabel:
			return Tree_node(label, None, None, None, True)

		if len(instances) < self.m:
			return Tree_node(default_label, None, None, None, True)

		self.candidate_splits(instances, attributes)
		attr, attr_count = self.max_info_gain(instances, attributes) 
		value = attributes[attr]


		tree = Tree_node(default_label, attr, attr_count, None, False);

		if type(value) == float: # numeric 
			for v in range(2):
				if v == 0:
					sub_instances = [inst for inst in instances if inst[attr] <= value ] 
				else:
					sub_instances = [inst for inst in instances if inst[attr] > value ] 

				if len(sub_instances) == 0:
					sub_tree = Tree_node(default_label, None, None, None, True)
				else:
					tmp_label = self.get_default_label(sub_instances)
					if tmp_label == None:
						tmp_label = tree.label
					sub_tree = self.build_tree(sub_instances, attributes.copy(), tmp_label) 

				sub_tree.parent_attribute_value = value
				tree.children.append(sub_tree)

		else: # nominal

			del attributes[attr]
			for v in value:
				sub_instances = [inst for inst in instances if inst[attr] == v ] 

				if len(sub_instances) == 0:
					sub_tree = Tree_node(default_label, None, None, None, True)
				else:
					tmp_label = self.get_default_label(sub_instances)
					if tmp_label == None:
						tmp_label = tree.label
					sub_tree = self.build_tree(sub_instances, attributes.copy(), tmp_label) 

				sub_tree.parent_attribute_value = v
				tree.children.append(sub_tree)

		return tree

	def get_default_label(self, instances):
		label_count = [0, 0]
		cls_values = self.cls_values
		for inst in instances:
			if cls_values[0] == inst['class']:
				label_count[0] += 1
			else:
				label_count[1] += 1

		if label_count[0] > label_count[1]:
			return cls_values[0]
		elif label_count[0] < label_count[1]:
			return cls_values[1]
		else:
			return None # tie
		


	def candidate_splits(self, instances, attributes):

		for attr in attributes:
			value = attributes[attr]
			
			if type(value) != list: # numeric
				## calculating threshold
				instances.sort(key = lambda k: k[attr])
				
				subsets = OrderedDict()
				for inst in instances:
					if inst[attr] not in subsets.keys():
						subsets[inst[attr]] = []
					subsets[inst[attr]].append(inst['class'])

				keys = subsets.keys()
				attributes[attr] = []
				for j in range(1, len(keys)):
					
					if subsets[keys[j-1]][0] == subsets[keys[j]][0] and \
					all(x == subsets[keys[j-1]][0] for x in subsets[keys[j-1]]) and \
					all(x == subsets[keys[j]][0] for x in subsets[keys[j]]):
						pass
					else:
						threshold = (keys[j-1] + keys[j]) / 2.0
						attributes[attr].append(threshold)
				
				if len(attributes[attr]) == 0:
					attributes[attr] = [keys[0]]


				

	def max_info_gain(self, instances, attributes):
		cls_values = self.cls_values
		attr_count = {}
		label_count = {}
		info_gains = {}
		max_gain = 0
		max_attr = attributes.keys()[0]

		for inst in instances:
			if inst['class'] not in label_count.keys():
				label_count[inst['class']] = 1
			else:
				label_count[inst['class']] += 1

		class_H = self.entropy_H(label_count[cls_values[0]], 
			label_count[cls_values[1]])

		# initialize 
		for k, v in attributes.items():
			if all(type(x) == float for x in v): # numeric
				attr_count[k] = [[{c : 0 for c in cls_values} for i in range(2)] for x in range(len(v))]
			else: # nominal
				attr_count[k] = {i : {c : 0 for c in cls_values} for i in v}

		for k, v in attributes.items():
			for inst in instances:
				if all(type(x) == float for x in v):  # numeric
					for x in range(len(v)):
						if inst[k] <= v[x]:
							attr_count[k][x][0][inst['class']] += 1
						else:
							attr_count[k][x][1][inst['class']] += 1
				else: # nominal
					attr_count[k][inst[k]][inst['class']] += 1


			if all(type(x) == float for x in v):  # numeric

				info_gains[k], attributes[k], partition = 0, v[0], attr_count[k][0]
				for x in range(len(v)):

					summation = sum([ (attr_count[k][x][attr][cls_values[0]] + attr_count[k][x][attr][cls_values[1]]) 
								* self.entropy_H(attr_count[k][x][attr][cls_values[0]], attr_count[k][x][attr][cls_values[1]]) 
								for attr in range(2)])
					
					local_info_gain = class_H - summation / len(instances)
					if local_info_gain >= info_gains[k]:
						info_gains[k], attributes[k], partition = local_info_gain, v[x], attr_count[k][x]
				
				attr_count[k] = partition

			else: # nominal

				summation = sum([ (attr_count[k][attr][cls_values[0]] + attr_count[k][attr][cls_values[1]]) 
							* self.entropy_H(attr_count[k][attr][cls_values[0]], attr_count[k][attr][cls_values[1]]) 
							for attr in v])
				
				local_info_gain = class_H - summation / len(instances)
				info_gains[k] = local_info_gain
			if max_gain < info_gains[k]:
				max_gain = info_gains[k]
				max_attr = k

		#print len(instances), attributes['thalach'], attr_count['thalach'], '\n\n'
		return max_attr, attr_count[max_attr]

	def entropy_H(self, *args):
		if sum(args) == 0:
			return 0
		return sum([-1.0 * x / sum(args) * math.log(1.0 * x / sum(args)) / math.log(2) 
			if x != 0 else 0 for x in args])

	def classify(self, tree, inst):
		while not tree.terminal:
			v = inst[tree.attribute] 
			if type(v) == float:
				if v <= tree.children[0].parent_attribute_value:
					tree = tree.children[0]
				else:
					tree = tree.children[1]
			else:
				for child in tree.children:
					if v == child.parent_attribute_value:
						tree = child
						break
		
		return tree.label


	def print_tree(self, node, k):
		cls_values = self.cls_values
		node_name = '|\t' * k
		node_name += '%s '%node.attribute
		if node.terminal:
			return
		c = 0 
		for child in node.children:
			value_name = ''
			
			value = child.parent_attribute_value
			if type(value) == float: 
				if c == 0:
					value_name += '<= %.6f'%value
				else:
					value_name += '> %.6f'%value
			else:
				value_name += '= ' + value
			
			if 	type(value) == float: 
				value_name += ' [%d %d]'%(node.attr_count[c][cls_values[0]], node.attr_count[c][cls_values[1]])
			else:
				value_name += ' [%d %d]'%(node.attr_count[value][cls_values[0]], node.attr_count[value][cls_values[1]])
			
			c += 1
			if child.terminal:
				value_name += ': %s'%child.label
			print node_name + value_name + '\r'
			self.print_tree(child, k+1)


