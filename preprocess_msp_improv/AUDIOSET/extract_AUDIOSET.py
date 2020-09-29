import os

import tensorflow

import numpy
import argparse

from scipy.io import wavfile

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('--model_file', default='vggish_model.ckpt')
parser.add_argument('--input_path', default='../../features/MSP-Improv/data')
parser.add_argument('--output_path', default='../../features/MSP-Improv/acoustic_features_VGGish')
args = parser.parse_args()

audio_path = args.input_path
acoustic_features_path = args.output_path
cnt = 0
for folder in os.listdir(audio_path):
	for sentence in os.listdir(os.path.join(audio_path, folder)):
		for scenario in os.listdir(os.path.join(audio_path, folder, sentence)):
			for audio in os.listdir(os.path.join(audio_path, folder, sentence, scenario)):
				if audio[-4:] == '.wav':
					wav_name = os.path.join(audio_path, folder, sentence, scenario, audio)
					wav_rate, wav_samples = wavfile.read(wav_name)
					if len(wav_samples) < wav_rate:
						wav_samples = numpy.pad(wav_samples, (0, wav_rate - len(wav_samples)), 'constant')

					samples = vggish_input.waveform_to_examples(wav_samples, wav_rate)

					with tensorflow.Graph().as_default(), tensorflow.Session() as session:
						vggish_slim.define_vggish_slim(training=False)
						vggish_slim.load_vggish_slim_checkpoint(session, args.model_file)
					
						samples_tensor = session.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
						features_tensor = session.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)

						[features] = session.run([features_tensor], feed_dict={samples_tensor: samples})

					output_file = os.path.join(acoustic_features_path, folder, sentence, scenario)
					os.makedirs(output_file, exist_ok=True)

					numpy.save(os.path.join(output_file, audio[:-4]), features)
					cnt += 1
					if cnt % 200 == 0:
						print(cnt)
print(cnt) #8438
