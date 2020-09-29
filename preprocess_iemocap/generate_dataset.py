import os
import csv
import pickle

def discretization(label):
	if label <= 2.75:
		label = 0
	elif label <= 3.25:
		label = 1
	else:
		label = 2

	return label

annotation_path = '../features/IEMOCAP/raw_data'
csv_path = '../dataset/IEMOCAP/all.csv'

annotations = []
a_labels = []
cnt = 0
for session in os.listdir(annotation_path):
	for annotation_file_name in os.listdir(os.path.join(annotation_path, session, 'dialog/EmoEvaluation')):
		if 'txt' in annotation_file_name:
			annotation_file = open(os.path.join(annotation_path, session, 'dialog/EmoEvaluation', annotation_file_name), 'r')
			line = annotation_file.readline()
			line = annotation_file.readline()
			lastline = line
			line = annotation_file.readline()
			while line:
				if lastline == '\n' and line != '\n':
					line = line[:-1].replace('\t', ' ').split(' ')
					v = float(line[5][1:-1])
					a = float(line[6][:-1])
					d = float(line[7][:-1])

					a = discretization(a)
					v = discretization(v)
					d = discretization(d)

					emotion_label = line[4]
					sentence_name = line[3]
					annotations.append([sentence_name, emotion_label, v, a, d])
					a_labels.append(a)
					cnt += 1

				lastline = line
				line = annotation_file.readline()

print(cnt) # 10039
print(a_labels.count(0), a_labels.count(1), a_labels.count(2)) # 3348 2569 4122

root_path = '../features'
domain_path = 'IEMOCAP'
visual_features_path = 'visual_features_resnet152_cropped_normalized'
acoustic_features_path = 'acoustic_features_MFB_normalized'
lexical_features_path = 'lexical_features_normalized'

with open(csv_path, 'w', newline='') as csvfile:
	writer = csv.writer(csvfile, delimiter=',', quotechar='|')
	writer.writerow([	'file_name_list', 'speakers',
						'visual_features', 'acoustic_features', 'lexical_features',
						'emotion_labels', 'v_labels', 'a_labels', 'd_labels'])

	cnt = 0
	for annotation in annotations:
		sentence_name = annotation[0]
		speaker = sentence_name[-4] + sentence_name[3:5]
		dialog_name = sentence_name[:-5]
		visual_features = os.path.join(root_path, domain_path, visual_features_path, 'Session'+sentence_name[4], dialog_name, sentence_name + '.npy')
		acoustic_features = os.path.join(root_path, domain_path, acoustic_features_path, 'Session'+sentence_name[4], dialog_name, sentence_name + '.npy')
		lexical_features = os.path.join(root_path, domain_path, lexical_features_path, 'Session'+sentence_name[4], dialog_name, sentence_name + '.npy')
		writer.writerow([sentence_name, speaker, visual_features, acoustic_features, lexical_features, annotation[1], annotation[2], annotation[3], annotation[4]])
