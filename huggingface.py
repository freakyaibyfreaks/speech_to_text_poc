import jovian

# Import necessary library
# For managing audio file
import librosa

#Importing Pytorch
import torch

#Importing Wav2Vec tokenizer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# Reading taken audio clip
import IPython.display as display
display.Audio("taken_clip.wav", autoplay=True)

# Loading the audio file
audio, rate = librosa.load("taken_clip.wav", sr = 16000)
print(rate)

# Importing Wav2Vec pretrained model
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Loading the audio file
audio, rate = librosa.load("taken_clip.wav", sr = 16000)

# Taking an input value
input_values = tokenizer(audio, return_tensors = "pt").input_values

# Storing logits (non-normalized prediction values)
logits = model(input_values).logits

# Storing predicted id's
prediction = torch.argmax(logits, dim = -1)

# Passing the prediction to the tokenzer decode to get the transcription
transcription = tokenizer.batch_decode(prediction)[0]

# Printing the transcription
print(transcription)

jovian.commit()