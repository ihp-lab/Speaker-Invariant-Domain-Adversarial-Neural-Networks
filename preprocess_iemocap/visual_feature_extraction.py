import os
import PIL
import torch
import models
import argparse
import numpy as np

from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num', default='cuda:0', help='GPU device')
parser.add_argument('--session', default='Session1')
opt = parser.parse_args()
device = torch.device(opt.gpu_num)

model = models.resnet152(pretrained=True, progress=False, opt=opt)
for param in model.parameters():
	param.requires_grad = False
model.to(device)

feature_extractor = models.FeatureExtractor(model)
for param in feature_extractor.parameters():
	param.requires_grad = False
feature_extractor.to(device)

transform_list = [	transforms.Resize(256),
					transforms.CenterCrop(224),
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

transform = transforms.Compose(transform_list)

frame_path = '../features/IEMOCAP/data_cropped'
feature_path = '../features/IEMOCAP/visual_features_resnet152_cropped'
session = opt.session

cnt = 0
for dialog in os.listdir(os.path.join(frame_path, session)):
	for sentence in os.listdir(os.path.join(frame_path, session, dialog)):
		file_name = os.path.join(feature_path, session, dialog, sentence+'.npy')
		if os.path.exists(file_name):
			cnt += 1
			feature_list = np.load(file_name)
			print(cnt, sentence, len(feature_list), len(feature_list[0]))
			continue
		feature_list = []
		for image_name in sorted(os.listdir(os.path.join(frame_path, session, dialog, sentence))):
			image = PIL.Image.open(os.path.join(frame_path, session, dialog, sentence, image_name))
			image = transform(image)
			shape = image.shape
			image = image.view(1, shape[0], shape[1], shape[2])
			image = image.to(device)

			feature = feature_extractor(image)
			feature = feature.view(-1).cpu().numpy().tolist()
			feature_list.append(feature)

		output_file = os.path.join(feature_path, session, dialog)
		os.makedirs(output_file, exist_ok=True)
		np.save(output_file + '/' + sentence, feature_list)
		cnt += 1
		print(cnt, sentence, len(feature_list), len(feature_list[0]))

print(cnt) # 10039
