import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import tensorflow 
import seaborn as sns 
import os 

from glob import glob
from itertools import cycle

import librosa as lr
import librosa.display
import IPython.display as ipd
import speech_recognition as sr
import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Inicio grabación...")

frames=[]
seconds = 3
for i in range(0, int(RATE/CHUNK * seconds)):
    data = stream.read(CHUNK)
    frames.append(data)
    
print("Fin de grabación")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open("entrada3.wav", 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()


data_dir='./Audios_test'
audio_files= glob(data_dir + './*wav')
lista=[]



audio_files
for file in os.listdir('Audios_test/'):
    print(file)



data_dir='./Audios_test'
audio_files= glob(data_dir + './*wav')
lista=[]
  

class AudioDataFixed: 
  def __init__(self):
    self.y = []
    self.d = []
    self.S_db=[]
    self.meanstft=[]
    self.y_trimmed=[]
    self.s=[]
    self.promedio=[]
    self.meany=[]
    self.meand=[]
    self.meanS_db=[]
    self.meany_trimmed=[]
    self.means=[]
    
    

  def getAudioDataFrame(self):
    return {'Y' : self.y, 'Stft': self.meanstft, "Amplitud":self.S_db, "Trimmed":self.y_trimmed, "mel_frequency":self.s}

  def meandataframe(self):
    return{'Y':self.meany,'Stft':self.promedio,"Amplitud":self.meanS_db, "Trimmed":self.meany_trimmed,"mel_frequency":self.means}
   

testAudios = AudioDataFixed()

for i in audio_files:
    #print('i:',i)
    y, sr=lr.load(i,sr=11025,offset=0.0,duration=2.0)                      ## direcion, url, array
    D = lr.stft(y) ## Short time fourier transform
    S_db= lr.amplitude_to_db(np.abs(D), ref=np.max) ## Amplitud a decibles(usado en tranformadas de audio conmunmente)
    s=librosa.feature.melspectrogram(y=y, sr=sr,n_mels=128)
    y_trimmed, _ =librosa.effects.trim(y, top_db=15)
    magnitud=abs(D)
    testAudios.y.append(y)
    testAudios.d.append(D)
    testAudios.S_db.append(S_db)
    testAudios.y_trimmed.append(y_trimmed)
    testAudios.s.append(s)
    testAudios.means.append(np.mean(s))
    testAudios.meany.append(np.mean(y))
    testAudios.meand.append(abs(D))
    testAudios.meanS_db.append(np.mean(S_db))
    testAudios.promedio.append
    testAudios.meany_trimmed.append(np.mean(y_trimmed))
    testAudios.meanstft.append(magnitud)
    testAudios.promedio.append(np.mean(magnitud))
    

    


dfa=pd.DataFrame(testAudios.getAudioDataFrame())
def valor_salida():
  lista=[]
  for i in range(0,151):
    if(i>=100):
      salida=1
      lista.append(salida)
    elif(i<100):
      salida=0
      lista.append(salida)
  return lista

dfb=pd.DataFrame({"Output":valor_salida()})

dfc=pd.DataFrame(testAudios.meandataframe())
dfc

b=1
output= b
dfc["Output"]= output
x=dfc.iloc[:,:5].values
x
y=dfc.iloc[:,5].values
y
from keras.utils import np_utils

nclasses = 2

Y_test = np_utils.to_categorical(y,nclasses)



from tensorflow.keras.models import model_from_json 

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
 
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# se cargan los pesos (weights) en el nuevo modelo
loaded_model.load_weights("model.h5")
print("Modelo cargado desde el PC")
# se evalua el modelo cargado con los datos de los test
loaded_model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
score = loaded_model.evaluate(x,Y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

entrada_x=loaded_model.predict(x)
entrada_x


valor_final=entrada_x[0][1]
valor_final



salida_USB=0
if (valor_final>=0.50):
    salida_USB=1
else:
    salida_USB=0

salida_USB



import serial,time



ser=serial.Serial('COM6',9600,timeout=1)
time.sleep(2)
if(salida_USB==1):
    ser.write(b'P')
    print("calculando....")
    print("voz reconocida..... Andrew")
else:
    ser.write(b'N')
    print("calculando.....")
    print("voz desconocida")


