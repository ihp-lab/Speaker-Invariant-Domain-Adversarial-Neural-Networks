import os

video_path = '../features/MSP-Improv/data'
frame_path = '../features/MSP-Improv/frames_30'

cnt = 0
for session in os.listdir(video_path):
	for sentence in os.listdir(os.path.join(video_path, session)):
		for scenario in os.listdir(os.path.join(video_path, session, sentence)):
			for video in os.listdir(os.path.join(video_path, session, sentence, scenario)):
				if video[-4:] == '.avi':
					input_path = os.path.join(video_path, session, sentence, scenario, video)
					output_file = os.path.join(frame_path, session, sentence, scenario, video[:-4])
					os.makedirs(output_file, exist_ok=True)
					cmd = 'ffmpeg -i '+ input_path + ' -vf fps=30 ' + output_file + '/' + video[:-4] + '_%05d.jpg'
					os.system(cmd)
					cnt += 1

print(cnt) # 8438
