import deepspeech

model_file_path = 'deepspeech-0.6.0-models/output_graph.pbmm'

beam_width = 500

model = deepspeech.Model(model_file_path, beam_width)

lm_file_path = 'deepspeech-0.6.0-models/lm.binary'
trie_file_path = 'deepspeech-0.6.0-models/trie'
lm_alpha = 0.75
lm_beta = 1.85
model.enableDecoderWithLM(lm_file_path, trie_file_path, lm_alpha, lm_beta)

import wave
filename = 'audio/8455-210777-0068.wav'
w = wave.open(filename, 'r')
rate = w.getframerate()
frames = w.getnframes()
buffer = w.readframes(frames)
print(rate)

print(model.sampleRate())

print(type(buffer))


import numpy as np
data16 = np.frombuffer(buffer, dtype=np.int16)
print(type(data16))

text = model.stt(data16)
print(text)