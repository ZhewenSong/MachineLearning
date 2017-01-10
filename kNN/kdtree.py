# -*- coding: utf-8 -*-
class kdnode:
	def __init__(self, key, feature, threshold, instance, left, right):
		self.key = key
		self.feature = feature # axis, x or y, 0 or 1
		self.threshold = threshold
		self.instance = instance # coordinate		
		self.left = left
		self.right = right


root = kdnode('f', 0, 6, [6,3], None, None)
root.left = kdnode('c', 1, 10, [5,10], None, None)
root.right = kdnode('h', 1, 5, [12,5], None, None)
root.left.left = kdnode('e', 1, 4, [2,4], None, None)
root.left.right = kdnode('b', 0, 3, [2,8], None, None)
root.right.left = kdnode('g', 0, 9, [9,2], None, None)
root.right.right = kdnode('i', 0, 10, [10,10], None, None)
root.left.left.right = kdnode('d', 1, 8, [2,8], None, None)
root.left.right.left = kdnode('a', 1, 11, [2,11], None, None)
root.right.right.right = kdnode('j', 1, 11.5, [13, 11.5], None, None)


def NN(x):
	PQ = [] # minimizing priority queue
	best_dist =  1000 # smallest distance seen so far
	dist = 1000
	best_node = root
	node = None
	PQ.append([root, 0])
	while len(PQ) > 0:
		PQ.sort(key = lambda k: k[1], reverse=True)

		if node == None:
			output = 'None, '
		else:
			output = '%s, '%node.key
		output += '%.4f, %.4f, %s, %s'%(dist, best_dist, best_node.key, 
			[[pq[0].key, pq[1]] for pq in PQ[::-1]])
		print output
		[node, bound] = PQ.pop()
		if bound >= best_dist:
			return best_node.key # nearest neighbor found
		dist = ((x[0]-node.instance[0])**2+(x[1]-node.instance[1])**2)**0.5
		if dist < best_dist:
			best_dist = dist
			best_node = node
		if x[node.feature] - node.threshold > 0:
			if node.left != None:
				PQ.append([node.left, x[node.feature] - node.threshold])
			if node.right != None:
				PQ.append([node.right, 0])
		else:
			if node.left != None:
				PQ.append([node.left, 0])
			if node.right != None:
				PQ.append([node.right, node.threshold - x[node.feature]])
	return best_node.key

print NN([7,10])