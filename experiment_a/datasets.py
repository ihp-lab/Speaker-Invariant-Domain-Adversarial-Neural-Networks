"""
Loads data
"""

import os
import PIL
import torch
import numpy as np
import pandas
import sampler

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class TemporalDataset(Dataset):
	def __init__(self, dataset_file_path, file_name_list):
		dataset_file = pandas.read_csv(dataset_file_path, index_col=0)
		dataset_subset = dataset_file.loc[file_name_list]

		self.file_name_list = file_name_list
		self.visual_features = dataset_subset.visual_features.to_dict()
		self.acoustic_features = dataset_subset.acoustic_features.to_dict()
		self.lexical_features = dataset_subset.lexical_features.to_dict()
		self.emotion_labels = dataset_subset.emotion_labels.to_dict()

		self.a_labels = dataset_subset.a_labels.to_dict()
		self.v_labels = dataset_subset.v_labels.to_dict()
		self.d_labels = dataset_subset.d_labels.to_dict()

		self.speakers = dataset_subset.speakers.to_dict()

	def __getitem__(self, idx):
		file_name = self.file_name_list[idx]

		visual_feature = np.load(self.visual_features[file_name])
		acoustic_feature = np.load(self.acoustic_features[file_name])
		lexical_feature = np.load(self.lexical_features[file_name])
		emotion_label = self.emotion_labels[file_name]

		v_label = self.v_labels[file_name]
		a_label = self.a_labels[file_name]
		d_label = self.d_labels[file_name]

		speaker = self.speakers[file_name]

		return visual_feature, acoustic_feature, lexical_feature, emotion_label, v_label, a_label, d_label, speaker

	def __len__(self):

		return len(self.file_name_list)

	def __getlabel__(self, idx):
		file_name = self.file_name_list[idx]
		a_label = self.a_labels[file_name]

		return a_label

def padding(features_tmp):
	features_dim = features_tmp[0].shape[1]
	lengths = [feature.shape[0] for feature in features_tmp]

	features = torch.zeros((len(features_tmp), max(lengths), features_dim)).float()
	for i, feature in enumerate(features_tmp):
		end = lengths[i]
		features[i, :end, :] = torch.FloatTensor(feature[:end, :])

	return features, torch.LongTensor(lengths)

def collate_fn_temporal_dataset(data):
	visual_features_tmp, acoustic_features_tmp, lexical_features, emotion_labels, v_labels, a_labels, d_labels, speakers = zip(*data)

	visual_features, visual_lengths = padding(visual_features_tmp)
	acoustic_features, acoustic_lengths = padding(acoustic_features_tmp)

	lexical_features = torch.FloatTensor(lexical_features)
	emotion_labels = torch.LongTensor(emotion_labels)

	v_labels = torch.LongTensor(v_labels)
	a_labels = torch.LongTensor(a_labels)
	d_labels = torch.LongTensor(d_labels)

	speakers = torch.LongTensor(speakers)

	return visual_features, visual_lengths, acoustic_features, acoustic_lengths, lexical_features, emotion_labels, v_labels, a_labels, d_labels, speakers

def get_temporal_dataset(dataset_file_path, file_name_list, batch_size, balance, shuffle, workers_num, collate_fn):
	dataset = TemporalDataset(dataset_file_path=dataset_file_path, file_name_list=file_name_list)

	if balance:
		dataloader = DataLoader(dataset=dataset,
								batch_size=batch_size,
								sampler=sampler.ImbalancedDatasetSampler(dataset),
								num_workers=workers_num,
								collate_fn=collate_fn,
								pin_memory=False)
	else:
		dataloader = DataLoader(dataset=dataset,
								batch_size=batch_size,
								shuffle=shuffle,
								num_workers=workers_num,
								collate_fn=collate_fn,
								pin_memory=False)

	return dataloader

def get_loaders_temporal_dataset(dataset_file_path, file_name_list, loader_type, opt):
	if loader_type == 'train':
		dataloader = get_temporal_dataset(	dataset_file_path=dataset_file_path,
											file_name_list=file_name_list,
											batch_size=opt.batch_size,
											balance=True,
											shuffle=True,
											workers_num=opt.workers_num,
											collate_fn=collate_fn_temporal_dataset)
	else:
		dataloader = get_temporal_dataset(	dataset_file_path=dataset_file_path,
											file_name_list=file_name_list,
											batch_size=opt.batch_size,
											balance=False,
											shuffle=False,
											workers_num=opt.workers_num,
											collate_fn=collate_fn_temporal_dataset)

	return dataloader
