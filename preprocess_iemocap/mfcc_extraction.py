import os
import librosa
import numpy as np

audio_path = '../features/IEMOCAP/data'
acoustic_features_path = '../features/IEMOCAP/acoustic_features_MFCC'

cnt = 0
for session in os.listdir(audio_path):
	for dialog in os.listdir(os.path.join(audio_path, session, 'sentences/wav')):
		if 'Ses' in dialog:
			for audio in os.listdir(os.path.join(audio_path, session, 'sentences/wav', dialog)):
				if audio[-4:] == '.wav':
					input_path = os.path.join(audio_path, session, 'sentences/wav', dialog, audio)
					y, sr = librosa.load(input_path)
					mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
					mfcc_delta = librosa.feature.delta(mfcc)
					mfcc_delta_2 = librosa.feature.delta(mfcc_delta)
					mfcc = np.concatenate((mfcc, mfcc_delta, mfcc_delta_2), axis=0) # 39 * T
					mfcc = np.transpose(mfcc) # T * 39

					output_file = os.path.join(acoustic_features_path, session, dialog)
					os.makedirs(output_file, exist_ok=True)
					np.save(os.path.join(output_file, audio[:-4]), mfcc)

					cnt += 1

print(cnt) # 10039
