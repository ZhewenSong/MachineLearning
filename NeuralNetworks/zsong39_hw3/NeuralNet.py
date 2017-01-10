from Parser import Data_Set
import math
from collections import OrderedDict
import numpy as np


class NNLearn:
	def __init__(self, rate, hno, epoch, train_set, test_set):
		self.rate = rate
		self.hno = hno 
		self.epoch = epoch
		self.train_set = train_set
		self.test_set = test_set
		self.attribute = train_set.attribute
		self.label = train_set.label
		self.ino = sum([len(v) if type(v) == list else 1 for k,v in self.attribute.items()])
		if self.hno == 0:
			self.wih = 0.02*np.random.rand(self.ino + 1, 1) - 0.01 # size: (ino + 1) x 1
		else:
			self.wih = 0.02*np.random.rand(self.ino + 1, self.hno) - 0.01 # size: (ino + 1) x hno
			self.who = 0.02*np.random.rand(self.hno + 1, 1) - 0.01 # size: (hno + 1) x 1
		#self.who = 0.02*np.random.rand(self.hno + 1, 2) - 0.01 # size: (hno + 1) x 2
		self.mu, self.sigma = self.standardize()

	def encode(self, inst):
		x = np.array([], dtype=float) 
		for k, v in self.attribute.items():
			if type(v) == list: # one-of-k encoding
				x = np.append(x, [1 if v[i] == inst[k] else 0 for i in range(len(v))]) 
			else:
				x = np.append(x, (inst[k] - self.mu[k])/self.sigma[k])
		x = np.append(x, 1)  # bias at the last unit
		return np.array([x])

	def standardize(self):
		mu = {}
		sigma = {}
		D = len(self.train_set.instances)
		for k, v in self.attribute.items():
			if type(v) != list:
				mu[k] = 1.0 * np.array([inst[k] for inst in self.train_set.instances]).sum() / D
				sigma[k] = np.sqrt(1.0 * np.array([(inst[k] - mu[k]) ** 2 for inst in self.train_set.instances]).sum() / D)

		return mu, sigma


	def get_output(self, inst, wih, who):
		layeri = self.encode(inst) # size: 1 x (ino + 1)
		net =  np.dot(layeri, wih)
		if self.hno == 0:
			layero = 1 / (1 + np.exp(-net.item()))  # size: 1 x 1
			return layeri, layero
		layerh = np.array([np.append(1 / (1 + np.exp(-net)), 1)]) # size: 1 x (hno + 1)
		net =  np.dot(layerh, who)
		layero = 1 / (1 + np.exp(-net.item()))  # size: 1 x 1
		return layeri, layerh, layero

	def learn(self):
		np.random.shuffle(self.train_set.instances)
		for e in range(1, self.epoch+1):
			correct = 0
			entropy = 0
			for i, inst in enumerate(self.train_set.instances):
				y = 0 if self.label[0] == inst['class'] else 1
				if self.hno == 0:
					layeri, layero = self.get_output(inst, self.wih, None)
					deltao = y - layero # size: 1 x 1
					dwih = self.rate * np.transpose(layeri) * deltao # size: (ino + 1) x 1
					self.wih += dwih
				else:
					layeri, layerh, layero = self.get_output(inst, self.wih, self.who)
					#### 1
					deltao = y - layero # size: 1 x 1  
					### 2
					deltah = layerh * (1 - layerh) * deltao * np.transpose(self.who) # size: 1 x (hno + 1) $ 
					### 3
					dwho = self.rate * np.transpose(layerh) * deltao # size: (hno + 1) x 1
					### 4
					dwih = self.rate * np.dot(np.transpose(layeri), np.array([deltah[0, 1:]])) # size: (ino + 1) x hno
					self.wih += dwih
					self.who += dwho
				if (layero > 0.5 and y == 1) or (layero < 0.5 and y == 0):
					correct += 1

				entropy += -y*np.log(layero)-(1-y)*np.log(1-layero)
			
			print "%d\t%.4f\t%d\t%d"%(e, entropy, correct, len(self.train_set.instances) - correct)

		return 1.0 * correct / len(self.train_set.instances)

	def classify(self):
		correct = 0
		data = []
		for inst in self.test_set.instances:
			y = 0 if self.label[0] == inst['class'] else 1
			if self.hno == 0:
				layeri, layero = self.get_output(inst, self.wih, None)
			else:
				layeri, layerh, layero = self.get_output(inst, self.wih, self.who)
			predicted = self.label[0] if layero < 0.5 else self.label[1]
			actual = inst['class']
			if predicted == actual:
				correct += 1
			print "%.4f\t%s\t%s"%(layero, predicted, actual)
			data.append([y, layero])
		print "%d\t%d"%(correct, len(self.test_set.instances) - correct)
		return 1.0 * correct / len(self.test_set.instances), np.array(data)

