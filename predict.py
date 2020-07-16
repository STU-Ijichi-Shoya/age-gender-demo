import warnings
warnings.simplefilter('ignore')

import tensorflow as tf
from tensorflow import keras
from efficientnet.tfkeras import EfficientNetB1

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

import tensorflow.keras.backend as K
import numpy

## save the GPU-memory 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")

K.set_learning_phase(0)
model=None

def load_model(model_path):
    global model
    model=keras.models.load_model(model_path)

""" rezised ndarray """
def recognize_r(F:numpy.ndarray):
    F=F.astype("float32")
    F/=255.0
    F=F.reshape((1,200,200,3))
    re=model.predict(F)
    gender="F" if re[1][0] >0.5 else "M"
    age=re[0][0]
    
    return gender,re[1][0]*100,age
