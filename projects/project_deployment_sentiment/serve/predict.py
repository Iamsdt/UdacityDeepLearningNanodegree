import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from model import LSTMClassifier

from utils import review_to_words, convert_and_pad



def model_fn(model_dir):
    """
    Load the PyTorch model from the `model_dir` directory
    """

    # Begin loading model:
    print("Loading model: Beginning...\n")

    # First, load the parameters used to create the model:
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)
    print("*** Model info: {}".format(model_info))

    # Determine the device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("*** Device: {}".format(device))

    # Construct the model:
    model = LSTMClassifier(model_info['embedding_dim'],
                           model_info['hidden_dim'],
                           model_info['vocab_size'])

    # Load the store model parameters:
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Load the saved word_dict:
    word_dict_path = os.path.join(model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'rb') as f:
        model.word_dict = pickle.load(f)

    # Move to evaluation mode:
    model.to(device).eval()

    # Print built model:
    print("*** Model:\n{}".format(model))

    # End loading model:
    print("\nLoading model: Done...")

    # Return model:
    return model



def input_fn(serialized_input_data, content_type):
    """
    Deserialize the input data
    """

    # Begin deserializing:
    print("Deserializing the input data...")

    # Check content type:
    if content_type == "text/plain":
        data = serialized_input_data.decode('utf-8')
        print("Done.")
        # Return deserialized data:
        return data

    # Content type not supported:
    raise Exception("Requested unsupported ContentType in content_type: {}".format(content_type))



def output_fn(prediction_output, accept):
    """
    Serialize the output data
    """

    # Begin serializing:
    print("Serializing the generated output...")

    # Perform serializing:
    serialized_prediction_output = str(prediction_output)
    print("Done.")

    # Return serialized prediction output:
    return serialized_prediction_output



def predict_fn(input_data, model):
    print('Inferring sentiment of input data.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model.word_dict is None:
        raise Exception('Model has not been loaded properly, no word_dict.')
    
    # TODO: Process input_data so that it is ready to be sent to our model.
    #       You should produce two variables:
    #         data_X   - A sequence of length 500 which represents the converted review
    #         data_len - The length of the review
    
    data_X ,data_len = convert_and_pad(model.word_dict, review_to_words(input_data))

    # Using data_X and data_len we construct an appropriate input tensor. Remember
    # that our model expects input data of the form 'len, review[500]'.
    data_pack = np.hstack((data_len, data_X))
    data_pack = data_pack.reshape(1, -1)
    
    data = torch.from_numpy(data_pack)
    data = data.to(device)
    model = model.to(device)

    # Make sure to put the model into evaluation mode
    model.eval()

    # TODO: Compute the result of applying the model to the input data. The variable `result` should
    #       be a numpy array which contains a single integer which is either 1 or 0
    
    # move to the model
    output_data = model.forward(data)

    # Transform output into a numpy array which contains a single integer, 1 or 0:
    if torch.cuda.is_available():
        # NumPy doesn't support CUDA:
        output_data.to('cpu')
        
        
    result = int(np.round(output_data.detach().numpy()))

    return result
