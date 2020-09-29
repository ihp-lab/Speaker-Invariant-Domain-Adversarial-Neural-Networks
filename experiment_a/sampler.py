"""
Samples elements randomly for imbalanced dataset
"""

import torch
import torch.utils.data
import torchvision

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

	def __init__(self, dataset, indices=None, num_samples=None):
		# if indices is not provided, all elements in the dataset will be considered
		self.indices = list(range(len(dataset))) \
			if indices is None else indices

		# if num_samples is not provided, draw `len(indices)` samples in each iteration
		self.num_samples = len(self.indices) \
			if num_samples is None else num_samples

		# distribution of classes in the dataset
		label_to_count = {}
		for idx in self.indices:
			label = self._get_label(dataset, idx)
			if label in label_to_count:
				label_to_count[label] += 1
			else:
				label_to_count[label] = 1

		# weight for each sample
		weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]

		self.weights = torch.DoubleTensor(weights)

	def __iter__(self):
		return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

	def __len__(self):
		return self.num_samples

	def _get_label(self, dataset, idx):
		return dataset.__getlabel__(idx)
