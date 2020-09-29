import csv
import os

people_dict = dict()
emotion_dict = dict()

people_dict['M01'] = 12
people_dict['M02'] = 13
people_dict['M03'] = 14
people_dict['M04'] = 15
people_dict['M05'] = 16
people_dict['F01'] = 17
people_dict['F02'] = 18
people_dict['F03'] = 19
people_dict['F04'] = 20
people_dict['F05'] = 21

emotion_dict['ang'] = 0
emotion_dict['sad'] = 1
emotion_dict['hap'] = 2
emotion_dict['oth'] = 3
emotion_dict['neu'] = 4
emotion_dict['xxx'] = 5
emotion_dict['fru'] = 6
emotion_dict['sur'] = 7
emotion_dict['fea'] = 8
emotion_dict['exc'] = 9
emotion_dict['dis'] = 10

def write_csv(file_name, input_lines):
    print(file_name)
    with open(file_name, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for row in input_lines:
            wr.writerow(row)

with open('../dataset/IEMOCAP/all.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')

    lines = []
    for row in csv_reader:
        lines.append(row)

    for row in lines[1:]:
        row[1] = people_dict[row[1]]
        row[5] = emotion_dict[row[5]]

    # select people to cross-validation set
    for selected_person in range(5):
        path = '../dataset/IEMOCAP/' + str(selected_person)
        os.makedirs(path, exist_ok=True)
        print(path)
        all_lines = []
        all_lines.append(lines[0])
        train_lines = []
        train_lines.append(lines[0])
        test_lines = []
        test_lines.append(lines[0])
        for row in lines[1:]:
            if row[1] == selected_person+12 or row[1] == selected_person+5+12:
                test_lines.append(row)
            else:
                train_lines.append(row)
            all_lines.append(row)
        # save to csv
        write_csv('../dataset/IEMOCAP/dataset.csv', all_lines)
        write_csv(path+'/train.csv', train_lines)
        write_csv(path+'/test.csv', test_lines)
