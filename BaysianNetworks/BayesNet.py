from Parser import Data_Set
import math
from collections import OrderedDict
import numpy as np


class NetNode:
	def __init__(self, index):
		self.index = index
		self.children = []
		self.parents = []

class Network:
	def __init__(self, node):
		self.repr = ''
		self.node = node

	def __repr__(self):
		def printNet(node, space):
			self.repr += space * '*' + str(node.index) + '\n'
			for c in node.children:
				printNet(c, space + 4)

		printNet(self.node, 0)
		return self.repr

	def __getitem__(self, index):
		self.rnode = None
		def getNode(node, index):
			if node.index == index:
				self.rnode = node
				return
			for c in node.children:
				getNode(c, index)
			return
		
		getNode(self.node, index)
		return self.rnode


class BNLearn:
	def __init__(self, train_set, test_set, attribute, label, option):
		self.train_set = train_set
		self.test_set = test_set
		self.option = option
		self.attribute = attribute
		self.variables = self.attribute.keys() + ['class']
		self.label = label
		self.values = self.attribute.copy()
		self.values['class'] = self.label
		self.instances = np.array([[inst[attr] for attr in self.attribute] for inst in self.train_set])
		self.instances = np.hstack((self.instances, [[inst['class']] for inst in self.train_set]))
		self.node_parents = [[] for i in range(len(self.variables))]
		self.prob_table_one_var = dict()  # P(X1)
		self.prob_table_one_var_one_cond = dict()  # P(X1 | C1)
		# P(X1, X2 | C1)  C1 = 'class'  X1 and X2 are symmetric
		self.prob_table_two_var_one_cond = dict()  
		# P(X1 | C1, C2)  C2 = 'class'
		self.prob_table_one_var_two_cond = dict()  
		# P(X1, X2, C1)   C1 = 'class'  X1 and X2 are symmetric
		self.prob_table_three_var = dict()         
		self.prob_call = 0


	def create_prob_table(self): 


		if self.option == 'n':

			J1, C1 = len(self.attribute.keys()), 'class'
			for c1 in self.values[C1]:
				self.prob_table_one_var['%s=%s'%(C1,c1)] = self._probability([[J1, c1]])

			for I1, X1 in enumerate(self.variables):
				for x1 in self.values[X1]:
					self.prob_table_one_var_one_cond['%s=%s'%(X1,x1)] = dict()
					J1, C1 = len(self.attribute.keys()), 'class'
					if X1 != C1:
						for c1 in self.values[C1]:
							self.prob_table_one_var_one_cond['%s=%s'%(X1,x1)]['%s=%s'%(C1,c1)] = \
							self._probability([[I1, x1]], cond_entry=[[J1, c1]])

		if self.option == 't':

			# print 'Creating prob_table_one_var'
			for I1, X1 in enumerate(self.variables):
				for x1 in self.values[X1]:
					self.prob_table_one_var['%s=%s'%(X1,x1)] = self._probability([[I1, x1]])
			# print self.prob_call

			# print 'Creating prob_table_one_var_one_cond'
			for I1, X1 in enumerate(self.variables):
				for x1 in self.values[X1]:
					self.prob_table_one_var_one_cond['%s=%s'%(X1,x1)] = dict()
					for J1, C1 in enumerate(self.variables):
						if X1 != C1:
							for c1 in self.values[C1]:
								self.prob_table_one_var_one_cond['%s=%s'%(X1,x1)]['%s=%s'%(C1,c1)] = \
								self._probability([[I1, x1]], cond_entry=[[J1, c1]])
			# print self.prob_call

			# print 'Creating prob_table_two_var_one_cond'
			for I1, X1 in enumerate(self.variables):
				for x1 in self.values[X1]:
					self.prob_table_two_var_one_cond['%s=%s'%(X1,x1)] = dict()
			for I1, X1 in enumerate(self.variables):
				for x1 in self.values[X1]:
					for I2 in range(I1+1, len(self.variables)):
						X2 = self.variables[I2]
						for x2 in self.values[X2]:
							self.prob_table_two_var_one_cond['%s=%s'%(X1,x1)]['%s=%s'%(X2,x2)] = dict()
							self.prob_table_two_var_one_cond['%s=%s'%(X2,x2)]['%s=%s'%(X1,x1)] = dict()
							J1, C1 = len(self.attribute.keys()), 'class'
							if X1 != C1:
								for c1 in self.values[C1]:
									self.prob_table_two_var_one_cond['%s=%s'%(X1,x1)]['%s=%s'%(X2,x2)]['%s=%s'%(C1,c1)] = \
									self._probability([[I1, x1], [I2, x2]], cond_entry=[[J1, c1]])
									self.prob_table_two_var_one_cond['%s=%s'%(X2,x2)]['%s=%s'%(X1,x1)]['%s=%s'%(C1,c1)] = \
									self.prob_table_two_var_one_cond['%s=%s'%(X1,x1)]['%s=%s'%(X2,x2)]['%s=%s'%(C1,c1)]
			# print self.prob_call

			# print 'Creating prob_table_one_var_two_cond'
			for I1, X1 in enumerate(self.variables):
				for x1 in self.values[X1]:
					self.prob_table_one_var_two_cond['%s=%s'%(X1,x1)] = dict()
					for J1, C1 in enumerate(self.variables):
						if X1 != C1:
							for c1 in self.values[C1]:
								self.prob_table_one_var_two_cond['%s=%s'%(X1,x1)]['%s=%s'%(C1,c1)] = dict()
								J2, C2 = len(self.attribute.keys()), 'class'
								if X1 != C2:
									for c2 in self.values[C2]:
										self.prob_table_one_var_two_cond['%s=%s'%(X1,x1)]['%s=%s'%(C1,c1)]['%s=%s'%(C2,c2)] = \
										self._probability([[I1, x1]], cond_entry=[[J1, c1], [J2, c2]])
			# print self.prob_call

			# print 'Creating prob_table_three_var'
			for I1, X1 in enumerate(self.variables):
				for x1 in self.values[X1]:
					self.prob_table_three_var['%s=%s'%(X1,x1)] = dict()
			for I1, X1 in enumerate(self.variables):
				for x1 in self.values[X1]:
					for I2 in range(I1+1, len(self.variables)):
						X2 = self.variables[I2]
						for x2 in self.values[X2]:
							self.prob_table_three_var['%s=%s'%(X1,x1)]['%s=%s'%(X2,x2)] = dict()
							self.prob_table_three_var['%s=%s'%(X2,x2)]['%s=%s'%(X1,x1)] = dict()
							J1, C1 = len(self.attribute.keys()), 'class'
							if X1 != C1:
								for c1 in self.values[C1]:
									self.prob_table_three_var['%s=%s'%(X1,x1)]['%s=%s'%(X2,x2)]['%s=%s'%(C1,c1)] = \
									self._probability([[I1, x1], [I2, x2], [J1, c1]])
									self.prob_table_three_var['%s=%s'%(X2,x2)]['%s=%s'%(X1,x1)]['%s=%s'%(C1,c1)] = \
									self.prob_table_three_var['%s=%s'%(X1,x1)]['%s=%s'%(X2,x2)]['%s=%s'%(C1,c1)]
			# print self.prob_call


	def _probability(self, attr_entry, cond_entry=[]):
		'''
		attr_entry: [[attr1_index, attr1], [attr2_index, attr2], ...]
		cond_entry: [[cond1_index, cond1], [cond2_index, cond2], ...]
		'''
		attr, cond = len(attr_entry), len(cond_entry)
		N = len(self.instances)
		delta = np.zeros([N, attr + cond])
		for x, entry in enumerate(attr_entry + cond_entry):
			delta[:, [x]] = self.instances[:, [entry[0]]] == np.array([[entry[1]] for inst in self.instances])


		count = np.sum([np.prod([delta[:, [x]] for x in range(attr + cond)], axis=0)])
		extra = np.prod([len(self.values[self.variables[int(k)]]) for k in np.array(attr_entry)[:,0]]) 

		total = np.sum([np.prod([delta[:, [x]] for x in range(attr, attr+cond)], axis=0)]) if cond > 0 else N
		#print count, extra, total
		self.prob_call += 1
		return 1.0 * (count + 1) / (total + extra)

	def _probability_join(self, instance, l):
		'''
		instance = {attr1: val1, attr2: val2, ...}
		'''
		variables = [0 for i in range(len(self.attribute))]
		for i, k in enumerate(self.attribute.keys()):
			v = instance[k]
			parents = self.node_parents[i]
			if len(parents) == 1:  # root of the spanning tree, i.e. i == 0
				assert(i == 0) 
				parents_w_c = [[len(self.attribute.keys()), self.label[l]]]
				parents_wo_c = []
				variables[i] = self.prob_table_one_var_one_cond['%s=%s'%(k, v)]['%s=%s'%('class', self.label[l])] / \
							   self.prob_table_one_var['%s=%s'%(k, v)]			
			else:
				attr = self.variables[parents[0].index]
				variables[i] = self.prob_table_one_var_two_cond['%s=%s'%(k, v)]['%s=%s'%(attr, instance[attr])]['%s=%s'%('class', self.label[l])] / \
							   self.prob_table_one_var_one_cond['%s=%s'%(k, v)]['%s=%s'%(attr, instance[attr])]

		return np.prod(variables)

	def cond_mut_info(self, id1, id2):
		attr1, attr2 = self.variables[id1], self.variables[id2]
		return sum(
			   sum(
   			   sum( self.prob_table_three_var['%s=%s'%(attr1, val1)]['%s=%s'%(attr2, val2)]['%s=%s'%('class', self.label[l])]
					* np.log2(
						self.prob_table_two_var_one_cond['%s=%s'%(attr1, val1)]['%s=%s'%(attr2, val2)]['%s=%s'%('class', self.label[l])]
					  /	self.prob_table_one_var_one_cond['%s=%s'%(attr1, val1)]['%s=%s'%('class', self.label[l])]
					  / self.prob_table_one_var_one_cond['%s=%s'%(attr2, val2)]['%s=%s'%('class', self.label[l])]
			   		)
			   for val1 in self.attribute[attr1]) 
			   for val2 in self.attribute[attr2]) 
			   for l in range(2))

	def find_MST(self, incident_matrix):
		n = len(self.variables) - 1
		heap = OrderedDict({index: 0 for index in range(n)})
		edge = OrderedDict()
		while heap:
			index = 0
			maxw = 0
			for k, v in heap.items():
				if v > maxw:
					index, maxw = k, v
			del heap[index]
			for nb in heap.keys():
				if incident_matrix[index][nb] > heap[nb]:
					heap[nb] = incident_matrix[index][nb]
					edge[nb] = index
		root = NetNode(0)
		edge = edge.items()
		visted = [0 for e in range(len(edge))]
		queue = [root]
		head = NetNode(len(self.attribute.keys()))
		while queue:
			curr = queue.pop(0)
			head.children.append(curr)
			curr.parents.append(head)
			index = curr.index
			self.node_parents[index].append(head)
			e = 0
			for v1, v2 in edge:
				if index in [v1, v2] and not visted[e]:
					child = NetNode(v2) if index == v1 else NetNode(v1)						
					curr.children.append(child)
					child.parents.append(curr)			
					self.node_parents[child.index].append(curr)			
					queue.append(child)
					visted[e] = 1
				e += 1
		#print Network(root)
		return Network(head), Network(root)


	def learn(self):

		self.create_prob_table()
		
		if self.option == 't':
			v = len(self.attribute)
			edge_weight = np.zeros([v, v])
			for i in range(v):
				for j in range(i+1, v):
					edge_weight[i, j] = self.cond_mut_info(i, j)
					edge_weight[j, i] = edge_weight[i, j]

			head, root = self.find_MST(edge_weight)
			#return head, root


	def classify(self):
		accuracy = 0
		total = len(self.test_set)
		string = ''
		self.learn()
		if self.option == 'n':
			
			for attr in self.attribute:
				string += '%s class\n'%attr

		if self.option == 't':

			for index in range(len(self.attribute.keys())):
				string += '%s '%self.variables[index]
				if len(self.node_parents[index]) > 1:
					attr_index = self.node_parents[index][0].index
					string += '%s '%self.variables[attr_index]
				string += 'class\n'

		string += '\n'
		for inst in self.test_set:
			Py = [0, 0]
			for l in range(2):
				if self.option == 'n':
					Py[l] = self.prob_table_one_var['%s=%s'%('class', self.label[l])] * \
						np.prod([self.prob_table_one_var_one_cond['%s=%s'%(attr, inst[attr])]['%s=%s'%('class', self.label[l])]
						for attr in self.attribute])
				if self.option == 't':
					Py[l] = self.prob_table_one_var['%s=%s'%('class', self.label[l])] * self._probability_join(inst, l)
			
			predict = self.label[0] if Py[0] > Py[1] else self.label[1]
			actual = inst['class']
			if predict == actual:
				accuracy += 1
			string += '%s %s %.12f\n'%(predict, actual, 
				max(1.0 * Py[0] / (Py[0] + Py[1]), 1.0 * Py[1] / (Py[0] + Py[1]))) 
			
		string += '\n%s'%accuracy
		print string
		return accuracy


