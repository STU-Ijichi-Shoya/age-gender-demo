
import tensorflow as tf
from tensorflow import keras
from efficientnet.tfkeras import EfficientNetB1

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

import tensorflow.keras.backend as K
import numpy as np

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")
    
K.set_learning_phase(0)
model=None

def load_model(model_path="/home/oit-trial/age-gender-estimeter/weights/eff-face-rate.hdf5"):
    global model
    model=keras.models.load_model(model_path)


""" rezised ndarray """
def recognize_r(F:list):
    size=len(F)
    F=np.array(F)
    F=F.astype("float32")
    F/=255.0
    F=F.reshape((size,200,200,3))
    # print(F.shape)
    re=model.predict(F)
    # print(re.shape)
    gender_list=[];age_list=[];gen_rate_list=[]
    for i in range(size):
        v,g=re[0][i],re[1][i]
        gender_list.append("F" if g >0.5 else "M")
        age_list.append(v)
        gen_rate_list.append(g)
        
    return gender_list,gen_rate_list,age_list
