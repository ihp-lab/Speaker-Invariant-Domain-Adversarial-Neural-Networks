import os
import numpy as np
import scipy.io.wavfile as wav

from python_speech_features import fbank

audio_path = '../features/MSP-Improv/data'
acoustic_features_path = '../features/MSP-Improv/acoustic_features_MFB'

cnt = 0
for session in os.listdir(audio_path):
	for sentence in os.listdir(os.path.join(audio_path, session)):
		for scenario in os.listdir(os.path.join(audio_path, session, sentence)):
			for audio in os.listdir(os.path.join(audio_path, session, sentence, scenario)):
				if audio[-4:] == '.wav':
					input_path = os.path.join(audio_path, session, sentence, scenario, audio)
					(rate, sig) = wav.read(input_path)
					feat, energy = fbank(sig, samplerate=rate, winlen=0.025, winstep=0.01, nfilt=40, nfft=2048, winfunc=np.hamming)
					output_file = os.path.join(acoustic_features_path, session, sentence, scenario)
					os.makedirs(output_file, exist_ok=True)
					np.save(os.path.join(output_file, audio[:-4]), feat)

					cnt += 1
					if cnt % 200 == 0:
						print(cnt)

print(cnt) # 8438
