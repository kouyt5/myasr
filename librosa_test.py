import wave
import librosa
from scipy.io.wavfile import read

my_audio = "./data/84.wav"
wave_path = librosa.util.example_audio_file()
y,sr = librosa.load(my_audio,sr=None)
D = librosa.stft(y) 
S,_ = librosa.magphase(D) # 分离实数和虚数
S = librosa.amplitude_to_db(S) # 
S = librosa.feature.melspectrogram(y,sr,n_fft=1024,n_mels=64)
S_dB = librosa.power_to_db(S)
x = librosa.feature.inverse.mel_to_audio(S,sr=sr,n_fft=1024)
librosa.stft()
librosa.istft()
y = read(my_audio)
print("")