import os
import numpy as np

visual_features_path = '../features/IEMOCAP/visual_features_resnet152_cropped'
new_visual_features_path = '../features/IEMOCAP/visual_features_resnet152_cropped_normalized'

dimensions = 2048
male_mean_list = [0]*dimensions
female_mean_list = [0]*dimensions
male_variance_list = [0]*dimensions
female_variance_list = [0]*dimensions
male_dim_list = {}
female_dim_list = {}

for d in range(0, dimensions):
	male_dim_list[d] = []
	female_dim_list[d] = []

for session in os.listdir(visual_features_path):
	for dialog in os.listdir(os.path.join(visual_features_path, session)):
		print(dialog)
		for sentence in os.listdir(os.path.join(visual_features_path, session, dialog)):
			feature = np.load(os.path.join(visual_features_path, session, dialog, sentence))
			gender = sentence[-8]

			if gender == 'M':
				for d in range(0, dimensions):
					male_dim_list[d].append(feature[:,d].flatten().tolist())
			else:
				for d in range(0, dimensions):
					female_dim_list[d].append(feature[:,d].flatten().tolist())

	for d in range(0, dimensions):
		male_dim_list[d] = [item for sublist in male_dim_list[d] for item in sublist]
		m = np.mean(male_dim_list[d])
		v = np.std(male_dim_list[d])
		male_mean_list[d] = m
		male_variance_list[d] = v

	for d in range(0, dimensions):
		female_dim_list[d] = [item for sublist in female_dim_list[d] for item in sublist]
		m = np.mean(female_dim_list[d])
		v = np.std(female_dim_list[d])
		female_mean_list[d] = m
		female_variance_list[d] = v

	for dialog in os.listdir(os.path.join(visual_features_path, session)):
		for sentence in os.listdir(os.path.join(visual_features_path, session, dialog)):
			feature = np.load(os.path.join(visual_features_path, session, dialog, sentence))
			gender = sentence[-8]
			for t in range(0, feature.shape[0]):
				for d in range(0, dimensions):
					if gender == 'M':
						feature[t,d] = (feature[t,d] - male_mean_list[d]) / (male_variance_list[d]+1e-6)
					else:
						feature[t,d] = (feature[t,d] - female_mean_list[d]) / (female_variance_list[d]+1e-6)

			output_file = os.path.join(new_visual_features_path, session, dialog)
			os.makedirs(output_file, exist_ok=True)
			np.save(os.path.join(output_file, sentence[:-4]), feature)

	for d in range(0, dimensions):
		male_dim_list[d] = []
		female_dim_list[d] = []
