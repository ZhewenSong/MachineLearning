def classify(train_instances, test_inst, k, attributes, kls):
	attribute = list(attributes)[:-1]
	distances = [0] * len(train_instances)
	for i, train_inst in enumerate(train_instances):
		dist = (sum([ (train_inst[attribute[attr]] - test_inst[attribute[attr]])**2 
			for attr in range(len(attribute)) ])) ** 0.5
		distances[i] = [dist, train_inst[kls]]
	distances.sort(key = lambda x: x[0])
        if kls == 'response':
		return sum([distances[i][1] for i in range(k)]) * 1.0 / k
	else:
		classes = attributes['class']
		counters = [0 for i in classes]
		neighbors = [distances[i][1] for i in range(k)]
		for i in range(k):
			counters[classes.index(neighbors[i])] += 1
		return classes[counters.index(max(counters))]
			


