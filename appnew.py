import pickle
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
import numpy as np

# keras imports
from keras.layers import (Input, Dense, Conv1D, MaxPool1D, Dropout, GRU, LSTM, 
                          TimeDistributed, Add, Flatten, RepeatVector, Lambda, Concatenate)
from keras.models import Model, load_model
from keras.metrics import binary_crossentropy
from keras import initializers
import keras.backend as K

from flask import Flask,render_template,request
app = Flask(__name__)

# Load it back
model = load_model("solubility_model_8858.hdf5")

# get unique character set in all SMILES strings 
charset = ['NULL', 'PAD', '#', '(', ')', '+', '-', '/', '1', '2', '3', '4', '5', '6', '7', ':', '=', 'B', 'C', 'F', 'H', 'I', 'N', 'O', 'P', 'S', '[', '\\', ']', 'l', 'r']

#print(charset )
def generate_charset(full_char_list:list) -> list:
    '''
    Assumes full_char_list is a list of characters (e.g., ['c', 'c', '1']).
    Returns a sorted list of unique characters, with index zero as a NULL character, and a PAD character.
    '''
    unique_chars = set(''.join(full_char_list))
    charset = ['NULL', 'PAD'] + sorted(unique_chars)
    return charset
import numpy as np 
def smiles_to_onehots(smiles_strings:list,
                     unique_charset:list,
                     max_smiles_chars:int) -> np.array:
    one_hots = []
    charset_length = len(unique_charset)

    for smiles_string in smiles_strings:
        one_hot_smiles = np.zeros(shape=(max_smiles_chars, charset_length))
        for i in range(max_smiles_chars):
            one_hot_col = [0]*charset_length
            ind = None # Which index will we flip to be "one-hot"?
            
            if i < len(smiles_string):
                try:
                    ind = unique_charset.index(smiles_string[i])
                    # one_hot_col[unique_charset.index(char)] = 1
                except ValueError:
                    ind = 0 # Treat as NULL if out-of-vocab  
                    # one_hot_col[0] = 1 # Treat as NULL if out-of-vocab   
            else:
                ind = 1 # Add PAD as needed
            
            one_hot_col[ind] = 1
            one_hot_smiles[i,:] = one_hot_col
            
        one_hots.append(one_hot_smiles)
    return np.array(one_hots)    

@app.route("/")
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        smiles = request.form["smiles"]
    #print(smiles)
    max_char_set= 97
    #
    #
    # charset = 34
    predict_test_input = smiles_to_onehots([smiles], charset, max_char_set)
    solubility_prediction = model.predict(predict_test_input)
    #predOUT = predictSingle(smiles, model)
    #predOUT = round(predOUT, 5)

    return render_template('index.html', prediction_text = "The log S is {}".format(solubility_prediction))
    #return render_template('sub.html',resu= "The log S is {}".format(predOUT))  

if __name__ == "__main__":
    app.run(debug=True, port=5000)
