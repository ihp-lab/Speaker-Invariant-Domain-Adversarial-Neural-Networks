"""
Defines models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

class GradientReversalFunction(Function):
	"""
	Gradient Reversal Layer from:
	Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)

	Forward pass is the identity function. In the backward pass,
	the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
	"""

	@staticmethod
	def forward(ctx, x, lambda_):
		ctx.lambda_ = lambda_
		
		return x.clone()

	@staticmethod
	def backward(ctx, grads):
		lambda_ = ctx.lambda_
		lambda_ = grads.new_tensor(lambda_)
		dx = -lambda_ * grads

		return dx, None

class GradientReversal(torch.nn.Module):
	def __init__(self, lambda_=1):
		super(GradientReversal, self).__init__()
		
		self.lambda_ = lambda_

	def forward(self, x):

		return GradientReversalFunction.apply(x, self.lambda_)

class Flatten(nn.Module):
	def forward(self, input):
		
		return input.view(input.size(0), -1)

class Model(nn.Module):
	def __init__(self, opt):
		super(Model, self).__init__()

		self.acoustic_modality = opt.acoustic_modality
		self.visual_modality = opt.visual_modality
		self.lexical_modality = opt.lexical_modality

		self.acoustic_feature_dim = opt.acoustic_feature_dim
		self.visual_feature_dim = opt.visual_feature_dim
		self.lexical_feature_dim = opt.lexical_feature_dim

		self.conv_width_v = opt.conv_width_v
		self.conv_width_a = opt.conv_width_a
		self.kernel_size_v = opt.kernel_size_v
		self.kernel_size_a = opt.kernel_size_a

		self.max_pool_width = opt.max_pool_width

		self.rnn_layer_num_v = opt.rnn_layer_num_v
		self.rnn_layer_num_a = opt.rnn_layer_num_a
		self.rnn_width = opt.rnn_width

		self.linear_width_l = opt.linear_width_l

		self.linear_width = opt.linear_width

		self.dropout_rate = opt.dropout_rate

		self.conv1d_v1 = nn.Conv1d(	in_channels=opt.visual_feature_dim,
									out_channels=self.conv_width_v,
									kernel_size=self.kernel_size_v,
									padding=self.kernel_size_v-1)
		self.conv1d_v2 = nn.Conv1d(	in_channels=self.conv_width_v,
									out_channels=self.conv_width_v,
									kernel_size=self.kernel_size_v,
									padding=self.kernel_size_v-1)
		self.conv1d_v3 = nn.Conv1d(	in_channels=self.conv_width_v,
									out_channels=self.conv_width_v,
									kernel_size=self.kernel_size_v,
									padding=self.kernel_size_v-1)

		self.conv1d_a1 = nn.Conv1d(	in_channels=opt.acoustic_feature_dim,
									out_channels=self.conv_width_a,
									kernel_size=self.kernel_size_a,
									padding=self.kernel_size_a-1)
		self.conv1d_a2 = nn.Conv1d(	in_channels=self.conv_width_a,
									out_channels=self.conv_width_a,
									kernel_size=self.kernel_size_a,
									padding=self.kernel_size_a-1)
		self.conv1d_a3 = nn.Conv1d(	in_channels=self.conv_width_a,
									out_channels=self.conv_width_a,
									kernel_size=self.kernel_size_a,
									padding=self.kernel_size_a-1)

		self.maxpool = nn.MaxPool1d(self.max_pool_width)

		self.gru_v = nn.GRU(input_size=self.conv_width_v,
							num_layers=self.rnn_layer_num_v,
							hidden_size=self.rnn_width,
							batch_first=True)

		self.gru_a = nn.GRU(input_size=self.conv_width_a,
							num_layers=self.rnn_layer_num_a,
							hidden_size=self.rnn_width,
							batch_first=True)

		self.linear_l = nn.Linear(self.lexical_feature_dim, self.linear_width_l)

		self.batchnorm_v = nn.BatchNorm1d(self.rnn_width)
		self.batchnorm_a = nn.BatchNorm1d(self.rnn_width)
		self.batchnorm_l = nn.BatchNorm1d(self.linear_width_l)
		self.dropout = nn.Dropout(self.dropout_rate)

		width = 0
		if self.acoustic_modality:
			width += self.rnn_width
		if self.visual_modality:
			width += self.rnn_width
		if self.lexical_modality:
			width += self.linear_width_l
		self.linear_1 = nn.Linear(width, self.linear_width)
		self.linear_2 = nn.Linear(self.linear_width, 3)
		self.softmax = nn.Softmax(dim=1)

		self.relu = nn.ReLU()

	def forward_v(self, x_v):
		x = x_v
		x = torch.transpose(x, 1, 2)
		x = self.relu(self.maxpool(self.conv1d_v1(x)))
		x = self.relu(self.maxpool(self.conv1d_v2(x)))
		x = self.relu(self.maxpool(self.conv1d_v3(x)))
		x = torch.transpose(x, 1, 2)
		x, _ = self.gru_v(x)
		x = torch.transpose(x, 1, 2)
		x = F.adaptive_avg_pool1d(x,1)[:, :, -1]
		x = self.batchnorm_v(self.dropout(x))

		return x

	def forward_a(self, x_a):
		x = x_a
		x = torch.transpose(x, 1, 2)
		x = self.relu(self.maxpool(self.conv1d_a1(x)))
		x = self.relu(self.maxpool(self.conv1d_a2(x)))
		x = self.relu(self.maxpool(self.conv1d_a3(x)))
		x = torch.transpose(x, 1, 2)
		x, _ = self.gru_a(x)
		x = torch.transpose(x, 1, 2)
		x = F.adaptive_avg_pool1d(x,1)[:, :, -1]
		x = self.batchnorm_a(self.dropout(x))

		return x

	def forward_l(self, x_l):
		x = x_l
		x = self.relu(self.linear_l(x))
		x = self.batchnorm_l(self.dropout(x))

		return x

	def encoder(self, x_v, x_a, x_l):
		if self.visual_modality:
			x_v = self.forward_v(x_v)
		if self.acoustic_modality:
			x_a = self.forward_a(x_a)
		if self.lexical_modality:
			x_l = self.forward_l(x_l)

		if self.visual_modality:
			if self.acoustic_modality:
				if self.lexical_modality:
					x = torch.cat((x_v, x_a, x_l), 1)
				else:
					x = torch.cat((x_v, x_a), 1)
			else:
				if self.lexical_modality:
					x = torch.cat((x_v, x_l), 1)
				else:
					x = x_v
		else:
			if self.acoustic_modality:
				if self.lexical_modality:
					x = torch.cat((x_a, x_l), 1)
				else:
					x = x_a
			else:
				x = x_l
		return x

	def recognizer(self, x):
		x = self.relu(self.linear_1(x))
		x = self.softmax(self.linear_2(x))

		return x

	def forward(self, x_v, x_a, x_l):
		x = self.encoder(x_v, x_a, x_l)
		x = self.recognizer(x)

		return x

class DomainDiscriminator(nn.Module):
	def __init__(self, opt):
		super(DomainDiscriminator, self).__init__()

		self.acoustic_modality = opt.acoustic_modality
		self.visual_modality = opt.visual_modality
		self.lexical_modality = opt.lexical_modality

		self.rnn_width = opt.rnn_width
		self.linear_width_l = opt.linear_width_l
		self.linear_width = opt.linear_width

		self.grl = GradientReversal(opt.domain_weight)

		width = 0
		if self.acoustic_modality:
			width += self.rnn_width
		if self.visual_modality:
			width += self.rnn_width
		if self.lexical_modality:
			width += self.linear_width_l

		self.linear_1 = nn.Linear(width, self.linear_width)
		self.linear_2 = nn.Linear(self.linear_width, 2)
		self.softmax = nn.Softmax(dim=1)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.grl(x)
		x = self.relu(self.linear_1(x))
		x = self.softmax(self.linear_2(x))

		return x

class SpeakerDiscriminator(nn.Module):
	def __init__(self, opt):
		super(SpeakerDiscriminator, self).__init__()

		self.acoustic_modality = opt.acoustic_modality
		self.visual_modality = opt.visual_modality
		self.lexical_modality = opt.lexical_modality

		self.rnn_width = opt.rnn_width
		self.linear_width_l = opt.linear_width_l
		self.linear_width = opt.linear_width

		self.grl = GradientReversal(opt.subject_weight)

		width = 0
		if self.acoustic_modality:
			width += self.rnn_width
		if self.visual_modality:
			width += self.rnn_width
		if self.lexical_modality:
			width += self.linear_width_l
		self.linear_1 = nn.Linear(width, self.linear_width)
		self.linear_2 = nn.Linear(self.linear_width, 22)
		self.softmax = nn.Softmax(dim=1)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.grl(x)
		x = self.relu(self.linear_1(x))
		x = self.softmax(self.linear_2(x))

		return x
