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

from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score

warnings.filterwarnings('always')

# Reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(0)

def main():
	parser = argparse.ArgumentParser()

	# Names, paths, logs
	parser.add_argument('--logger_path', default='checkpoints/dann', help='relative path to log')
	parser.add_argument('--source_domain', default='', help='MSP-Improv or IEMOCAP')
	parser.add_argument('--target_domain', default='', help='MSP-Improv or IEMOCAP')
	parser.add_argument('--verbose', type=bool, default=False, help='True or False')

	# Data parameters
	parser.add_argument('--workers_num', type=int, default=4, help='number of workers for data loading')

	# Training and optimization
	parser.add_argument('--epochs_num', type=int, default=25, help='number of training epochs')
	parser.add_argument('--batch_size', type=int, default=256, help='size of a mini-batch')

	parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate')
	parser.add_argument('--domain_weight', type=float, default=3)

	# Modality
	parser.add_argument('--acoustic_modality', type=bool, default=True)
	parser.add_argument('--visual_modality', type=bool, default=True)
	parser.add_argument('--lexical_modality', type=bool, default=True)

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
	opt.target_domain = 'IEMOCAP'
	acc_1, uar_1, acc_std_1, uar_std_1 = domain_adaptation(opt)

	opt.source_domain = 'IEMOCAP'
	opt.target_domain = 'MSP-Improv'
	acc_2, uar_2, acc_std_2, uar_std_2 = domain_adaptation(opt)

	print(acc_1, ',', uar_1, ',', acc_2, ',', uar_2, ',', acc_1+uar_1+acc_2+uar_2, ',', acc_std_1, uar_std_1, acc_std_2, uar_std_2)

def domain_adaptation(opt):
	# Use specific GPU
	device = torch.device(opt.gpu_num)

	half_batch = opt.batch_size // 2
	opt.batch_size = half_batch

	# Dataloaders
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

	# Model, optimizer and loss function
	checkpoint = torch.load(os.path.join(	'checkpoints/bl', opt.source_domain, 'model.pth.tar'),
											map_location=device)

	emotion_recognizer = models.Model(opt)
	emotion_recognizer.load_state_dict(checkpoint['emotion_recognizer'])
	for param in emotion_recognizer.parameters():
		param.requires_grad = True
	emotion_recognizer.to(device)

	discriminator = models.DomainDiscriminator(opt)
	discriminator.apply(models.init_weights)
	for param in discriminator.parameters():
		param.requires_grad = True
	discriminator.to(device)

	optimizer = torch.optim.Adam(	list(emotion_recognizer.parameters())
									+list(discriminator.parameters()),
									lr=opt.learning_rate)

	criterion = torch.nn.CrossEntropyLoss()

	best_acc = 0.
	best_acc_std = 0.

	best_uar = 0.
	best_uar_std = 0.

	# Train and validate
	for epoch in range(opt.epochs_num):
		if opt.verbose:
			print('epoch: {}/{}'.format(epoch + 1, opt.epochs_num))

		batch_iterator, n_batches = get_batch_iterator(opt)

		domain_loss, domain_acc, train_loss, train_acc = train(	batch_iterator, n_batches,
																emotion_recognizer, discriminator,
																optimizer, criterion, device, opt)
		test_loss, test_acc, test_uar = test(test_loader, emotion_recognizer, criterion, device, opt)

		acc_list = []
		uar_list = []
		for i in range(folder_num):
			loader = test_loader_list[i]
			_, acc, uar = test(loader, emotion_recognizer, criterion, device, opt)
			acc_list.append(acc)
			uar_list.append(uar)
		
		acc_list = np.array(acc_list)
		uar_list = np.array(uar_list)

		acc_std = np.std(acc_list)
		uar_std = np.std(uar_list)

		if opt.verbose:
			print(	'domain_loss: {0:.5f}'.format(domain_loss),
					'domain_acc: {0:.3f}'.format(domain_acc),
					'train_loss: {0:.5f}'.format(train_loss),
					'train_acc: {0:.3f}'.format(train_acc),
					'test_loss: {0:.5f}'.format(test_loss),
					'test_acc: {0:.3f}'.format(test_acc),
					'test_uar: {0:.3f}'.format(test_uar))

		os.makedirs(os.path.join(opt.logger_path, opt.source_domain), exist_ok=True)

		model_file_name = os.path.join(opt.logger_path, opt.source_domain, 'checkpoint.pth.tar')
		state = {	'epoch': epoch+1, 'emotion_recognizer': emotion_recognizer.state_dict(),
					'discriminator' : discriminator.state_dict(), 'opt': opt}
		torch.save(state, model_file_name)

		if test_acc > best_acc and epoch >= 3:
			model_file_name = os.path.join(opt.logger_path, opt.source_domain, 'model.pth.tar')
			torch.save(state, model_file_name)

			best_acc = test_acc
			best_acc_std = acc_std

		if test_uar > best_uar and epoch >= 3:
			best_uar = test_uar
			best_uar_std = uar_std

	return best_acc, best_uar, best_acc_std, best_uar_std

def get_dataloader(dataset_file_path, loader_type, opt):
	# Data
	data = pandas.read_csv(dataset_file_path)
	file_name_list = data['file_name_list'].tolist()

	dataloader = datasets.get_loaders_temporal_dataset(	dataset_file_path,
														file_name_list,
														loader_type, opt)

	return dataloader

def get_batch_iterator(opt):
	source_dataset_file_path = os.path.join('../dataset', opt.source_domain, '0', 'train.csv')
	source_loader = get_dataloader(source_dataset_file_path, 'train', opt)

	target_dataset_file_path = os.path.join('../dataset', opt.target_domain, 'dataset.csv')
	target_loader = get_dataloader(target_dataset_file_path, 'train', opt)

	batches = zip(source_loader, target_loader)
	n_batches = min(len(source_loader), len(target_loader))

	return batches, n_batches

def train(batches, n_batches, model, discriminator, optimizer, criterion, device, opt):
	model.train()

	total_domain_loss = 0
	domain_acc = 0
	total_label_loss = 0
	label_acc = 0

	for i, train_data in enumerate(batches):
		(source_x_v, _, source_x_a, _, source_x_l, _, source_y_v, _, _, _), \
		(target_x_v, _, target_x_a, _, target_x_l, _, _, _, _, _) = train_data

		source_x_v = source_x_v.to(device)
		source_x_a = source_x_a.to(device)
		source_x_l = source_x_l.to(device)
		source_y_v = source_y_v.to(device)

		target_x_v = target_x_v.to(device)
		target_x_a = target_x_a.to(device)
		target_x_l = target_x_l.to(device)

		source_encoded_x = model.encoder(source_x_v, source_x_a, source_x_l)
		target_encoded_x = model.encoder(target_x_v, target_x_a, target_x_l)

		encoded_x = torch.cat([source_encoded_x, target_encoded_x])
		encoded_x = encoded_x.to(device)

		domain_y = torch.cat([	torch.ones(source_encoded_x.shape[0], dtype=torch.int64),
								torch.zeros(target_encoded_x.shape[0], dtype=torch.int64)]).to(device)
		label_y = source_y_v.to(device)

		domain_preds = discriminator(encoded_x).squeeze()
		label_preds = model.recognizer(source_encoded_x)

		domain_loss = criterion(domain_preds, domain_y)
		label_loss = criterion(label_preds, label_y)
		loss = domain_loss + label_loss

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		total_domain_loss += domain_loss.item()
		total_label_loss += label_loss.item()

		domain_preds = domain_preds.argmax(dim=1, keepdim=True)
		domain_acc += domain_preds.eq(domain_y.view_as(domain_preds)).sum().item() / len(domain_preds)

		label_preds = label_preds.argmax(dim=1, keepdim=True)
		label_acc += label_preds.eq(label_y.view_as(label_preds)).sum().item() / len(label_preds)

		if opt.verbose and i > 0 and i % int(n_batches / 10) == 0:
			print('.', flush=True, end='')

		if i >= n_batches:
			break

	domain_loss = total_domain_loss / n_batches
	domain_acc = domain_acc / n_batches
	label_loss = total_label_loss / n_batches
	label_acc = label_acc / n_batches

	return domain_loss, domain_acc, label_loss, label_acc

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
