import re
from collections import OrderedDict

class Data_Set:
	def __init__(self, inputfile):
		self.relation, self.attribute, self.instances = self.parse(inputfile)

	def parse(self, inputfile):

		with open(inputfile) as fd:
			contains = fd.readlines()

		attribute = OrderedDict()
		attribute_names = []
		instances = []
		for line in contains:
			if line[0] in ['%', '\n', '\r']: # comment or blank line
				pass
			else:
				if '@relation' in line.lower():
					relation = line.split()[1]
				elif '@attribute' in line.lower():
					words = filter(None, re.split('[@{ ,}\n\r\']', line))
					attribute_name = words[1]
					attribute_names.append(attribute_name)
					if len(words) > 3:
						attribute[attribute_name] = words[2:]
					else:
						attribute[attribute_name] = words[2]
				elif '@' not in line:
					attrs = filter(None, re.split('[ ,\n\r\']', line))
					inst = {}
					for i in range(len(attrs)):
						try:
							inst[attribute_names[i]] = float(attrs[i])
						except ValueError:
							inst[attribute_names[i]] = attrs[i]

					instances.append(inst)

		return (relation, attribute, instances)
