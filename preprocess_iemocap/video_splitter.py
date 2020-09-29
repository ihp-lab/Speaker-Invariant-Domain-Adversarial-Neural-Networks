import os

def get_start_end_time_from_lines(lines):
  start_end_time = []
  for line in lines:
    end_char_index = 0
    if line[0] == '[':
      for i in range(len(line)):
        if line[i] == ']':
          end_char_index = i
          break
      temp_line = line[1:end_char_index].split(' - ')
      start_end_time.append(temp_line)
  return start_end_time

def get_output_sentence_name_from_lines(lines):
  output_filenames = []
  for line in lines:
    if line[0] == '[':
      start_index = 0
      end_index = 0
      for i in range(len(line)):
        if line[i] == ']':
          start_index = i
        if start_index != 0 and line[i] == '[':
          end_index = i
          break
      temp_line = line[start_index+1:end_index].split('\t')[1]
      output_filenames.append(temp_line)
  return output_filenames

def create_dir(directory):
  os.makedirs(directory, exist_ok=True)

for session_num in range(1,6):
  sessions_dir = '../features/IEMOCAP/data/'
  video_dir = sessions_dir + 'Session'+ str(session_num) +'/dialog/avi/DivX'
  splitting_file_dir = sessions_dir + 'Session'+ str(session_num) +'/dialog/EmoEvaluation'

  # search split files
  split_file_list = []
  for entry in os.listdir(splitting_file_dir):
    if entry.startswith('S') and entry.endswith('.txt'):
      split_file_list.append(os.path.join(splitting_file_dir, entry))

  output_session_dir = sessions_dir + 'Session' + str(session_num) + '/sentences/avi'
  create_dir(output_session_dir)

  # loop dialog
  for split_file in split_file_list:
    with open(split_file, 'r') as f:
      lines = f.readlines()
      start_end_time = get_start_end_time_from_lines(lines)
      print(split_file)
      dialog_name = split_file.split('/')[-1].split('.')[0]
      video_file_path = video_dir + '/' + dialog_name + '.avi'
      sentences_name = get_output_sentence_name_from_lines(lines)
      dialog_path = output_session_dir + '/' + dialog_name

      create_dir(dialog_path)

      # loop sentence
      for i in range(len(start_end_time)):
        start = start_end_time[i][0]
        end = start_end_time[i][1]

        sentence_path = dialog_path + '/' + sentences_name[i]

        create_dir(sentence_path)

        cmd = 'ffmpeg -i '+ video_file_path + ' -ss ' + start + ' -to '+ end + ' -vf fps=30 ' + sentence_path + '/' + '%05d.jpg'
        os.system(cmd)
