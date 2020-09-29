"""
Tests models
"""

import os
import torch
import random
import pandas
import models
import datasets
import argparse
import itertools
import numpy as np

from sklearn.metrics import accuracy_score, recall_score

# Reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(0)

def main():
	parser = argparse.ArgumentParser()

	# Names, paths, logs
	parser.add_argument('--source_domain', default='', help='MSP-Improv or IEMOCAP')
	parser.add_argument('--target_domain', default='', help='MSP-Improv or IEMOCAP')

	# Data parameters
	parser.add_argument('--workers_num', type=int, default=0, help='number of workers for data loading')

	# Model
	parser.add_argument('--model', default='bl', help='model to test')

	# Modality
	parser.add_argument('--acoustic_modality', type=bool, default=True)
	parser.add_argument('--visual_modality', type=bool, default=True)
	parser.add_argument('--lexical_modality', type=bool, default=True)

	# Training and optimization
	parser.add_argument('--batch_size', type=int, default=256, help='size of a mini-batch')

	# Model parameters
	parser.add_argument('--visual_feature_dim', type=int, default=2048)
	parser.add_argument('--acoustic_feature_dim', type=int, default=40)
	parser.add_argument('--lexical_feature_dim', type=int, default=768)

	parser.add_argument('--conv_width_v', type=int, default=64, help='64 or 128')
	parser.add_argument('--conv_width_a', type=int, default=128, help='64 or 128')
	parser.add_argument('--kernel_size_v', type=int, default=2, help='2 or 3')
	parser.add_argument('--kernel_size_a', type=int, default=3, help='2 or 3')
	parser.add_argument('--max_pool_width', type=int, default=2)
	parser.add_argument('--rnn_layer_num_v', type=int, default=3, help='2 or 3')
	parser.add_argument('--rnn_layer_num_a', type=int, default=3, help='2 or 3')
	parser.add_argument('--rnn_width', type=int, default=32)
	parser.add_argument('--linear_width_l', type=int, default=32, help='32')
	parser.add_argument('--linear_width', type=int, default=64, help='32 or 64')
	parser.add_argument('--dropout_rate', type=float, default=0.3, help='0.3')

	# GPU
	parser.add_argument('--gpu_num', default='cuda:0', help='GPU device')

	opt = parser.parse_args()

	print('Training and validating models')
	for arg in vars(opt):
		print(arg + ' = ' + str(getattr(opt, arg)))

	opt.source_domain = 'MSP-Improv'
	opt.target_domain = 'IEMOCAP'
	test_one_domain(opt)

	opt.source_domain = 'IEMOCAP'
	opt.target_domain = 'MSP-Improv'
	test_one_domain(opt)

def test_one_domain(opt):
	# Use specific GPU
	device = torch.device(opt.gpu_num)

	# Dataloader
	test_dataset_file_path = os.path.join('../dataset', opt.target_domain, 'dataset.csv')
	test_loader = get_dataloader(test_dataset_file_path, 'test', opt)

	if opt.target_domain == 'MSP-Improv':
		folder_num = 6
	else:
		folder_num = 5
	
	test_loader_list = []
	for i in range(folder_num):
		dataset_file_path = os.path.join('../dataset', opt.target_domain, str(i), 'test.csv')
		loader = get_dataloader(dataset_file_path, 'test', opt)
		test_loader_list.append(loader)

	# Model and loss function
	emotion_recognizer = models.Model(opt)
	emotion_recognizer.to(device)

	criterion = torch.nn.CrossEntropyLoss()

	checkpoint = torch.load(os.path.join('checkpoints', opt.model, opt.source_domain, 'model.pth.tar'),
							map_location=device)
	emotion_recognizer.load_state_dict(checkpoint['emotion_recognizer'])

	test_loss, test_acc, test_uar = test(test_loader, emotion_recognizer, criterion, device)

	acc_list = []
	uar_list = []
	for i in range(folder_num):
		loader = test_loader_list[i]
		_, acc, uar = test(loader, emotion_recognizer, criterion, device)
		acc_list.append(acc)
		uar_list.append(uar)
	
	acc_list = np.array(acc_list)
	uar_list = np.array(uar_list)

	acc_std = np.std(acc_list)
	uar_std = np.std(uar_list)

	print(	'epoch: {}'.format(checkpoint['epoch']), 'test_loss: {0:.5f}'.format(test_loss),
			'test_acc: {0:.3f}'.format(test_acc), 'test_uar: {0:.3f}'.format(test_uar),
			'acc_std: {0:.3f}'.format(acc_std), 'uar_std: {0:.3f}'.format(uar_std))

def get_dataloader(dataset_file_path, loader_type, opt):
	# Data
	data = pandas.read_csv(dataset_file_path)
	file_name_list = data['file_name_list'].tolist()

	dataloader = datasets.get_loaders_temporal_dataset(	dataset_file_path,
														file_name_list,
														loader_type, opt)

	return dataloader

def test(test_loader, model, criterion, device):
	model.eval()

	running_loss = 0.
	running_acc = 0.

	with torch.no_grad():
		groundtruth = []
		prediction = []

		for i, test_data in enumerate(test_loader):
			visual_features, _, acoustic_features, _, lexical_features, _, v_labels, _, _, speakers = test_data

			visual_features = visual_features.to(device)
			acoustic_features = acoustic_features.to(device)
			lexical_features = lexical_features.to(device)

			labels = v_labels.to(device)

			predictions = model(visual_features, acoustic_features, lexical_features)
			loss = criterion(predictions, labels)

			running_loss += loss.item()

			groundtruth.append(labels.tolist())
			predictions = predictions.argmax(dim=1, keepdim=True)
			prediction.append(predictions.view_as(labels).tolist())

		test_loss = running_loss / len(test_loader)

		groundtruth = list(itertools.chain.from_iterable(groundtruth))
		prediction = list(itertools.chain.from_iterable(prediction))

		test_acc = accuracy_score(prediction, groundtruth)
		test_uar = recall_score(prediction, groundtruth, average='macro')

		return test_loss, test_acc, test_uar

if __name__ == '__main__':
	main()
