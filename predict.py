import numpy as np
import pickle
from utils import feature_extractor


def predict_gender(audio_path):
    try:
        feats = feature_extractor(audio_path)
        feats = np.expand_dims(feats, axis=0)

        scaler = pickle.load(open('scaler.pkl', 'rb'))
        feats = scaler.transform(feats)
        model = pickle.load(open('model.pkl', 'rb'))

        output = model.predict(feats)
        if output == 1:
            gen = 'Female'
        elif output == 0:
            gen = 'Male'
    except:
        print('ERROR')

    return gen
