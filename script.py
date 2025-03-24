#####################################################################################################################
#####                                                                                                           #####
#####                                            POSITIONAL ENCODING                                            #####
#####                                           Created on: 2025-03-23                                          #####
#####                                           Updated on: 2025-03-24                                          #####
#####                                                                                                           #####
#####################################################################################################################

#####################################################################################################################
#####                                                  PACKAGES                                                 #####
#####################################################################################################################

# Clear the whole environment
globals().clear()

# Load the libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.decomposition import PCA

# Set up the right directory
import os
os.chdir('C:/Users/Admin/Documents/Python Projects/_embedding')


#####################################################################################################################
#####                                     POSITIONAL ENCODING VISUALIZATION                                     #####
#####################################################################################################################

# Create the function to get the positional encodings
def positional_encoding(sequence_size, encoding_size):
    '''
    Compute a matrix with all the positional encodings

    Inputs:
    sequence_size -- maximum number of positions to be encoded (int), maximum length of the sentence
    encoding_size -- encoding size (int)

    Returns:
    pos_encoding -- matrix with the positional encodings, shape (1, sequence_size, encoding_size) 
    '''
    # Create column vector (2D array) from 1D array with values from 0 to sequence_size-1
    pos_matrix = np.arange(sequence_size)[:, np.newaxis]
    # Create row vector (2D array) from 1D array with values from 0 to encoding_size-1
    d_matrix = np.arange(encoding_size)[np.newaxis, :]

    # Define a matrix angle_rads of all the angles 
    i = d_matrix//2
    angle_rads = pos_matrix/(10000**((2*i)/encoding_size))

    # Apply sine function to even indices in the array
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
    # Apply cosine function to odd indices in the array
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    # Add another dimension (1st dimension, being the batch dimension)
    pos_encoding = angle_rads[np.newaxis, ...]
    
    # Transform pos_encoding to float
    pos_encoding_output = tf.cast(pos_encoding, dtype=tf.float32)

    return pos_encoding_output


# Define the embedding size
EMBEDDING_DIM = 100
# Define the maximum length of the sequence
MAX_SEQUENCE_LENGTH = 50
# Define the max number of words we keep to tokenize (we compute the frequency of each word appearing in our text and we keep the MAX_NB_WORDS most frequent that we transform to an index, the rest is transformed to OOV if we set up the option)
MAX_NB_WORDS = 64

# Determine the positional encodings
pos_encoding = positional_encoding(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)

# Get the shape of the positional encoding
pos_encoding.shape

# Plot the positional encodings
plt.pcolormesh(pos_encoding[0], cmap='RdBu')
plt.xlabel('d')
plt.xlim((0, EMBEDDING_DIM))
plt.ylabel('Position')
plt.colorbar()
plt.show()


# Determine the norm of a vector (and notice that whatever the pos, the norm is the same)
pos = np.random.randint(MAX_SEQUENCE_LENGTH)
tf.norm(pos_encoding[0,pos,:])

# Determine the difference between the norm of 2 vectors (and notice that whatever the pos, the difference is the same [as long as k is the same])
pos = np.random.randint(MAX_SEQUENCE_LENGTH)
k = 2
print(tf.norm(pos_encoding[0,pos,:] -  pos_encoding[0,pos + k,:]))

