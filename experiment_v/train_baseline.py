"""
Trains and validates models
"""

import os
import torch
import random
import pandas
import models
import warnings
import datasets
import argparse
import itertools
import numpy as np
import csv

from utils import EarlyStopping
from sklearn.metrics import accuracy_score, recall_score

warnings.filterwarnings('always')

# Reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random.seed(0)
np.random.seed(0)

def main():
	parser = argparse.ArgumentParser()

	# Names, paths, logs
	parser.add_argument('--logger_path', default='checkpoints/bl', help='relative path to log')
	parser.add_argument('--source_domain', default='', help='MSP-Improv or IEMOCAP')
	parser.add_argument('--verbose', type=bool, default=True, help='True or False')

	# Data parameters
	parser.add_argument('--workers_num', type=int, default=4, help='number of workers for data loading')
	parser.add_argument('--folder', default='')

	# Modality
	parser.add_argument('--acoustic_modality', type=bool, default=True)
	parser.add_argument('--visual_modality', type=bool, default=True)
	parser.add_argument('--lexical_modality', type=bool, default=True)

	# Training and optimization
	parser.add_argument('--epochs_num', type=int, default=50, help='number of training epochs')
	parser.add_argument('--batch_size', type=int, default=256, help='size of a mini-batch')
	parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate')
	parser.add_argument('--patience', type=int, default=5, help='early stopping patience')

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

	if opt.verbose:
		print('Training and validating models')
		for arg in vars(opt):
			print(arg + ' = ' + str(getattr(opt, arg)))

	opt.source_domain = 'MSP-Improv'
	train_one_folder(opt, str(0))
	opt.source_domain = 'IEMOCAP'
	train_one_folder(opt, str(0))

def train_one_folder(opt, folder):
	# Use specific GPU
	device = torch.device(opt.gpu_num)

	opt.folder = folder

	# Dataloaders
	train_dataset_file_path = os.path.join('../dataset', opt.source_domain, str(opt.folder), 'train.csv')
	train_loader = get_dataloader(train_dataset_file_path, 'train', opt)

	test_dataset_file_path = os.path.join('../dataset', opt.source_domain, str(opt.folder), 'test.csv')
	test_loader = get_dataloader(test_dataset_file_path, 'test', opt)

	# Model, optimizer and loss function
	emotion_recognizer = models.Model(opt)
	models.init_weights(emotion_recognizer)
	for param in emotion_recognizer.parameters():
		param.requires_grad = True
	emotion_recognizer.to(device)

	optimizer = torch.optim.Adam(emotion_recognizer.parameters(), lr=opt.learning_rate)
	lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)

	criterion = torch.nn.CrossEntropyLoss()

	best_acc = 0.
	best_uar = 0.
	es = EarlyStopping(patience=opt.patience)

	# Train and validate
	for epoch in range(opt.epochs_num):
		if opt.verbose:
			print('epoch: {}/{}'.format(epoch + 1, opt.epochs_num))

		train_loss, train_acc = train(	train_loader, emotion_recognizer,
										optimizer, criterion, device, opt)
		test_loss, test_acc, test_uar = test(	test_loader, emotion_recognizer,
												criterion, device, opt)

		if opt.verbose:
			print(	'train_loss: {0:.5f}'.format(train_loss), 'train_acc: {0:.3f}'.format(train_acc),
					'test_loss: {0:.5f}'.format(test_loss), 'test_acc: {0:.3f}'.format(test_acc),
					'test_uar: {0:.3f}'.format(test_uar))

		lr_schedule.step(test_loss)

		os.makedirs(os.path.join(opt.logger_path, opt.source_domain), exist_ok=True)

		model_file_name = os.path.join(opt.logger_path, opt.source_domain, 'checkpoint.pth.tar')
		state = {'epoch': epoch+1, 'emotion_recognizer': emotion_recognizer.state_dict(), 'opt': opt}
		torch.save(state, model_file_name)

		if test_acc > best_acc and epoch >= 3:
			model_file_name = os.path.join(opt.logger_path, opt.source_domain, 'model.pth.tar')
			torch.save(state, model_file_name)

			best_acc = test_acc

		if test_uar > best_uar and epoch >= 3:
			best_uar = test_uar

		if es.step(test_loss):
			break

	return best_acc, best_uar

def get_dataloader(dataset_file_path, loader_type, opt):
	# Data
	data = pandas.read_csv(dataset_file_path)
	file_name_list = data['file_name_list'].tolist()

	dataloader = datasets.get_loaders_temporal_dataset(	dataset_file_path,
														file_name_list,
														loader_type, opt)

	return dataloader

def train(train_loader, model, optimizer, criterion, device, opt):
	model.train()

	running_loss = 0.
	running_acc = 0.

	groundtruth = []
	prediction = []

	for i, train_data in enumerate(train_loader):
		visual_features, _, acoustic_features, _, lexical_features, _, v_labels, _, _, speakers = train_data

		visual_features = visual_features.to(device)
		acoustic_features = acoustic_features.to(device)
		lexical_features = lexical_features.to(device)

		labels = v_labels.to(device)

		optimizer.zero_grad()
		predictions = model(visual_features, acoustic_features, lexical_features)

		loss = criterion(predictions, labels)

		loss.backward()
		optimizer.step()

		running_loss += loss.item()

		groundtruth.append(labels.tolist())
		predictions = predictions.argmax(dim=1, keepdim=True)
		prediction.append(predictions.view_as(labels).tolist())

		if opt.verbose and i > 0 and int(len(train_loader) / 10) > 0 and i % (int(len(train_loader) / 10)) == 0:
			print('.', flush=True, end='')

	train_loss = running_loss / len(train_loader)

	groundtruth = list(itertools.chain.from_iterable(groundtruth))
	prediction = list(itertools.chain.from_iterable(prediction))

	train_acc = accuracy_score(prediction, groundtruth)

	return train_loss, train_acc

def test(test_loader, model, criterion, device, opt):
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
