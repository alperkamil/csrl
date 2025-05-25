from random import sample

class ReplayMemory:
	"""ring buffer with efficient random sampling. details at:
	https://stackoverflow.com/questions/40181284/how-to-get-random-sample-from-deque-in-python-3/40182242"""
	def __init__(self, max_size):
		self.buffer = [None] * max_size
		self.max_size = max_size
		self.index = 0
		self.size = 0

	def append(self, obj):
		self.buffer[self.index] = obj
		self.size = min(self.size + 1, self.max_size)
		self.index = (self.index + 1) % self.max_size

	def sample(self, batch_size):
		"""returns a uniformly random list of size batch_size"""
		indices = sample(range(self.size), batch_size)
		return [self.buffer[index] for index in indices]
	
	def getSize(self):
		return self.size