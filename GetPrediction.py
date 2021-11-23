#!/usr/bin/env python
# coding: utf-8

# ## Function to load model

# In[1]:


def load_model():

    # loading json and creating model
    from keras.models import model_from_json
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")
    print("Loaded model from disk")
    # evaluate loaded model on test data
    # loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    loaded_model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    # score = loaded_model.evaluate(x_testcnn, y_test, verbose=0)
    # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    
    return loaded_model


# ## Running model

# In[2]:


import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix


# In[3]:


from sklearn.preprocessing import LabelEncoder
data, sampling_rate = librosa.load('output_m_happy.wav')


# In[4]:


def get_prediction(file_received):

    # loading json and creating model
    import librosa
    import librosa.display
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from matplotlib.pyplot import specgram
    import keras
    from keras.preprocessing import sequence
    from keras.models import Sequential
    from keras.layers import Dense, Embedding
    from keras.layers import LSTM
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    # from keras.utils import to_categorical
    from keras.layers import Input, Flatten, Dropout, Activation
    from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
    from keras.models import Model
    from keras.callbacks import ModelCheckpoint
    from sklearn.metrics import confusion_matrix
    from keras.models import model_from_json
    
    from sklearn.preprocessing import LabelEncoder
    data, sampling_rate = librosa.load('output_m_happy.wav')

    import os
    import pandas as pd
    import librosa
    import glob 

    plt.figure(figsize=(15, 5))
    librosa.display.waveplot(data, sr=sampling_rate)
    
    livedf= pd.DataFrame(columns=['feature'])
    
    X, sample_rate = librosa.load(file_received, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
    print("Happy file = ", X)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
    featurelive = mfccs
    livedf2 = featurelive
    
    livedf2= pd.DataFrame(data=livedf2)
    livedf2 = livedf2.stack().to_frame().T
    livedf2

    twodim= np.expand_dims(livedf2, axis=2)
    
    loaded_model = load_model()

    livepreds = loaded_model.predict(twodim, 
                             batch_size=32, 
                             verbose=1)
    livepreds
    livepreds1=livepreds.argmax(axis=1)
    livepreds1

    liveabc = livepreds1.astype(int).flatten()
    print(liveabc)
    
    return liveabc[0]


# ## Label encoding 

# In[5]:


def encode_response(index):
    response = ""
    if index==0:
        response = 'female_angry'
    elif index==1:
        response = 'female_calm'
    elif index==2:
        response = 'female_fearful'
    elif index==3:
        response = 'female_happy'
    elif index==4:
        response = 'female_sad'
    elif index==5:
        response = 'male_angry'
    elif index==6:
        response = 'male_calm'
    elif index==7:
        response = 'male_fearful'
    elif index==8:
        response = 'male_happy'
    elif index==9:
        response = 'male_sad'
    else:
        response = "sentiment not recognized!"
    print(response)
    return response


# In[6]:


ALLOWED_EXTENSIONS = set(['wav'])


# In[7]:


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ## Creating REST API endpoint

# In[8]:


import flask
from flask import request, jsonify
import os

app = flask.Flask(__name__)
port = int(os.getenv("PORT"))

@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404

# A route to return sentiment for the input file/text from the user.
@app.route('/api/v1/resources/sentiments/all/', methods=['POST'])
def get_sentiment():
    file_received = request.files['testfile']
    response = {
        "predictedVoice": ""
    }
    index = get_prediction(file_received)
    print(index)
    response["predictedVoice"] = encode_response(index)
    print(response)
    return jsonify(response)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)


# In[9]:


# !pip3 freeze > requirements.txt

