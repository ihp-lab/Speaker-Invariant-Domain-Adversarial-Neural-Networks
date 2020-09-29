import os
import csv
import pickle

empty_sentence_path = '../features/MSP-Improv/empty_sentence.txt'

empty_sentences = []
file = open(empty_sentence_path, 'r')
line = file.readline()
while line:
	line = line[:-5]
	empty_sentences.append(line)
	line = file.readline()

def discretization(label):
	if label <= 2.75:
		label = 0
	elif label <= 3.25:
		label = 1
	else:
		label = 2

	return label

annotation_path = '../features/MSP-Improv/Evalution.txt'
csv_path = '../dataset/MSP-Improv/all.csv'

annotation_file = open(annotation_path, 'r')

annotations = []
tmp_list = []
line = annotation_file.readline()
while line:
	if line == '\n':
		annotations.append(tmp_list[0])
		tmp_list = []
		line = annotation_file.readline()
	else:
		tmp_list.append(line[:-1])
		line = annotation_file.readline()

annotation_dict = {}
for annotation in annotations:
	annotation = annotation.replace(' ', '').split(';')
	annotation[0] = annotation[0].replace('UTD', 'MSP')
	label = annotation[1]
	a = float(annotation[2][2:])
	v = float(annotation[3][2:])
	d = float(annotation[4][2:])

	a = discretization(a)
	v = discretization(v)
	d = discretization(d)

	annotation_dict[annotation[0]] = [label, v, a, d]

root_path = '../features'
domain_path = 'MSP-Improv'
visual_features_path = 'visual_features_resnet152_normalized'
acoustic_features_path = 'acoustic_features_MFB_normalized'
lexical_features_path = 'lexical_features_normalized'

with open(csv_path, 'w', newline='') as csvfile:
	writer = csv.writer(csvfile, delimiter=',', quotechar='|')
	writer.writerow([	'file_name_list', 'speakers',
						'visual_features', 'acoustic_features', 'lexical_features',
						'emotion_labels', 'v_labels', 'a_labels', 'd_labels'])

	cnt = 0
	for file_name, labels in annotation_dict.items():
		name = file_name[:-4].split('-')
		speaker = name[3]

		if os.path.join('session'+name[3][-1], name[2], name[4], file_name[:-4]) in empty_sentences:
			continue

		visual_features = os.path.join(root_path, domain_path, visual_features_path, 'session'+name[3][-1], name[2], name[4], file_name[:-4] + '.npy')
		acoustic_features = os.path.join(root_path, domain_path, acoustic_features_path, 'session'+name[3][-1], name[2], name[4], file_name[:-4] + '.npy')
		lexical_features = os.path.join(root_path, domain_path, lexical_features_path, 'session'+name[3][-1], name[2], name[4], file_name[:-4] + '.npy')
		writer.writerow([file_name, speaker, visual_features, acoustic_features, lexical_features, labels[0], labels[1], labels[2], labels[3]])
		cnt += 1

print(cnt) # 8166
