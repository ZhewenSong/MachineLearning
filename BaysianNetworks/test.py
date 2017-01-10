import random
instances = [random.randint(0,2**70) for i in range(3000)]
test = random.randint(0,2**70)
import time
start = time.time()
for i in range(30000):
	bit = [(2**34) , (2**54) , (2**12)]
	for j in range(3000):
		a = instances[j] ^ test
		a = (a%bit[0]) * (a%bit[1]) * (a%bit[2])
end = time.time()
print end - start