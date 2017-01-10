from Parser import Data_Set
import math
from collections import OrderedDict
import numpy as np


class NetNode:
	def __init__(self, key):
		self.key = key # attr
		self.children = []
		self.parents = []

class Network:
	def __init__(self, node):
		self.repr = ''
		self.node = node

	def __repr__(self):
		def printNet(node, space):
			self.repr += space * '-' + node.key + '\n'
			for c in node.children:
				printNet(c, space + 4)

		printNet(self.node, 0)
		return self.repr

	def __getitem__(self, key):
		self.rnode = None
		def getNode(node, key):
			if node.key == key:
				self.rnode = node
				return
			for c in node.children:
				getNode(c, key)
			return
		
		getNode(self.node, key)
		return self.rnode


class BNLearn:
	def __init__(self, train_set, test_set, option):
		self.train_set = train_set
		self.test_set = test_set
		self.option = option
		self.attribute = train_set.attribute
		self.label = train_set.label


	def _probability(self, instances, attribute, cond={}):
		'''
		attribute = {attr1: val1, attr2: val2, ...}
		cond = {attr1: val1, attr2: val2, ...}
		'''
		count = 0
		total = 0

		for inst in instances:
			found_cond = 1 if cond else 0
			for k, v in cond.items():
				if v != inst[k]:
					found_cond = 0
					break
			if found_cond:
				total += 1

			found_attr = 1
			for k, v in attribute.items():
				if v != inst[k]:
					found_attr = 0
					break
			if found_attr:
				if cond == {}:
					count += 1
				elif found_cond:
					count += 1
		
		extra = np.prod([2 if k == 'class' else len(self.attribute[k]) for k in attribute.keys()]) 
		
		if cond == {}:
			total = len(instances)

		#print count, extra, total
		return 1.0 * (count + 1) / (total + extra)

	def _probability_join(self, instances, root, attribute):
		'''
		attribute = {attr1: val1, attr2: val2, ...}
		'''
		variables = [0 for i in attribute.keys()]
		for i, k in enumerate(attribute.keys()):
			parents = {p.key: attribute[p.key] for p in root[k].parents}
			variables[i] = self._probability(instances, {k: attribute[k]}, cond=parents)
			print k, attribute[k], parents
		return np.prod(variables)

	def cond_mut_info(self, instances, attr1, attr2):
		return sum(
			   sum(
   			   sum( self._probability(instances, {attr1: val1, attr2: val2, 'class': self.label[l]})
					#self._probability(instances, {attr1: val1, attr2: val2}, cond={'class': self.label[l]})
					#* self._probability(instances, {'class': self.label[l]})
					* np.log2(
						self._probability(instances, {attr1: val1, attr2: val2}, cond={'class': self.label[l]}) 
					  /	self._probability(instances, {attr1: val1}, cond={'class': self.label[l]})
					  / self._probability(instances, {attr2: val2}, cond={'class': self.label[l]})
			   		)
			   for val1 in self.attribute[attr1]) 
			   for val2 in self.attribute[attr2]) 
			   for l in range(2))

	def find_MST(self, incident_matrix):
		vertices = self.attribute.keys()
		n = len(vertices)
		heap = OrderedDict({attr: 0 for attr in vertices})
		edge = OrderedDict()
		while heap:
			attr = vertices[0]
			maxw = 0
			for k, v in heap.items():
				if v > maxw:
					attr, maxw = k, v
			del heap[attr]
			index = vertices.index(attr)
			for nb in range(n):
				if nb != index:
					nb_attr = vertices[nb]
					if nb_attr in heap.keys(): 
						if incident_matrix[index][nb] > heap[nb_attr]:
							heap[nb_attr] = incident_matrix[index][nb]
							edge[nb_attr] = attr
		root = NetNode(vertices[0])
		edge = edge.items()
		visted = [0 for e in range(len(edge))]
		queue = [root]
		head = NetNode('class')
		while queue:
			curr = queue.pop(0)
			head.children.append(curr)
			curr.parents.append(head)
			key = curr.key
			e = 0
			for v1, v2 in edge:
				if key in [v1, v2] and not visted[e]:
					child = NetNode(v2) if key == v1 else NetNode(v1)						
					curr.children.append(child)
					child.parents.append(curr)						
					queue.append(child)
					visted[e] = 1
				e += 1
		#print Network(root)
		return Network(head), Network(root)


	def learn(self):
		instances = self.train_set.instances
		if self.option == 'n':
			Py = [0, 0]
			Py[0] = self._probability(instances, {'class': self.label[0]})
			Py[1] = self._probability(instances, {'class': self.label[1]})
			Px = OrderedDict()
			for attr in self.attribute:
				Px[attr] = OrderedDict()
				for val in self.attribute[attr]:
					Px[attr][val] = [ self._probability(instances, {attr: val}, cond={'class': self.label[0]}), 
									  self._probability(instances, {attr: val}, cond={'class': self.label[1]}) ]
			return Px, Py
		if self.option == 't':
			v = len(self.attribute)
			edge_weight = np.zeros([v, v])
			for i in range(v):
				for j in range(i+1, v):
					attr1, attr2 = self.attribute.keys()[i], self.attribute.keys()[j]
					edge_weight[i, j] = self.cond_mut_info(instances, attr1, attr2)
					edge_weight[j, i] = edge_weight[i, j]
			head, root = self.find_MST(edge_weight)
			return head, root


	def classify(self):
		accuracy = 0
		total = len(self.test_set.instances)
		string = ''
		if self.option == 'n':
			Px, Py0 = self.learn()
			for attr in self.attribute:
				string += '%s class\n'%attr

		if self.option == 't':
			head, root = self.learn()
			for attr in self.attribute:
				string += '%s '%attr
				curr = root[attr]
				if len(curr.parents) > 1:
					string += '%s '%curr.parents[0].key
				string += 'class\n'

		string += '\n'
		for inst in self.test_set.instances:
			Py = [0, 0]
			if self.option == 't':
				attribute = inst.copy()
				
			for l in range(2):
				if self.option == 'n':
					Py[l] = Py0[l] * np.prod([Px[attr][inst[attr]][l] for attr in self.attribute])
				if self.option == 't':
					attribute['class'] = self.label[l]
					Py[l] = self._probability_join(self.train_set.instances, head, attribute)
					Py[l] /= self._probability_join(self.train_set.instances, head, inst)
			
			predict = self.label[0] if Py[0] > Py[1] else self.label[1]
			actual = inst['class']
			if predict == actual:
				accuracy += 1
			string += '%s %s %.12f\n'%(predict, actual, 
				max(1.0 * Py[0] / (Py[0] + Py[1]), 1.0 * Py[1] / (Py[0] + Py[1]))) 
		string += '\n%s'%accuracy
		print string


