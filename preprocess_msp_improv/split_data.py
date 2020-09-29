import csv
import os

people_dict = dict()
emotion_dict = dict()
people_dict['M01'] = 0
people_dict['M02'] = 1
people_dict['M03'] = 2
people_dict['M04'] = 3
people_dict['M05'] = 4
people_dict['M06'] = 5
people_dict['F01'] = 6
people_dict['F02'] = 7
people_dict['F03'] = 8
people_dict['F04'] = 9
people_dict['F05'] = 10
people_dict['F06'] = 11

emotion_dict['A'] = 0
emotion_dict['S'] = 1
emotion_dict['H'] = 2
emotion_dict['O'] = 3
emotion_dict['N'] = 4
emotion_dict['X'] = 5

def write_csv(file_name, input_lines):
    print(file_name)
    with open(file_name, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for row in input_lines:
            wr.writerow(row)

with open('../dataset/MSP-Improv/all.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')

    lines = []
    for row in csv_reader:
        lines.append(row)

    for row in lines[1:]:
        row[1] = people_dict[row[1]]
        row[5] = emotion_dict[row[5]]

    # select people to cross-validation set
    for selected_person in range(6):
        path = '../dataset/MSP-Improv/' + str(selected_person)
        os.makedirs(path, exist_ok=True)
        print(path)
        all_lines = []
        all_lines.append(lines[0])
        train_lines = []
        train_lines.append(lines[0])
        test_lines = []
        test_lines.append(lines[0])
        for row in lines[1:]:
            if row[1] == selected_person or row[1] == selected_person+6:
                test_lines.append(row)
            else:
                train_lines.append(row)
            all_lines.append(row)
        # save to csv
        write_csv('../dataset/MSP-Improv/dataset.csv', all_lines)
        write_csv(path+'/train.csv', train_lines)
        write_csv(path+'/test.csv', test_lines)
