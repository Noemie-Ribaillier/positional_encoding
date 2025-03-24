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


#####################################################################################################################
#####                                       COMPARE POSITIONAL ENCODINGS                                        #####
#####################################################################################################################

# Compute the correlation between the positional encodings
tf_corr = tf.matmul(pos_encoding, pos_encoding, transpose_b=True)
# Transform to numpy array
np_corr = tf_corr.numpy()
# Take the 1st element to remove the batch dimensions
corr = np_corr[0]

# Plot the correlation between the positional encodings
plt.pcolormesh(corr, cmap='RdBu')
plt.xlabel('Position')
plt.xlim((0, MAX_SEQUENCE_LENGTH))
plt.ylabel('Position')
plt.colorbar()
plt.show()


# Compute the Euclidean distance between the positional encoding
# Fill the matrix with 0
eu = np.zeros((MAX_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH))
print(eu.shape)
# Iterate on the rows
for a in range(MAX_SEQUENCE_LENGTH):
    # Iterate on the columns
    for b in range(a + 1, MAX_SEQUENCE_LENGTH):
        eu[a, b] = tf.norm(tf.math.subtract(pos_encoding[0, a], pos_encoding[0, b]))
        # Euclidean distance is symmetric
        eu[b, a] = eu[a, b]

# Plot the Euclidean distance between the positional encoding
plt.pcolormesh(eu, cmap='RdBu')
plt.xlabel('Position')
plt.xlim((0, MAX_SEQUENCE_LENGTH))
plt.ylabel('Position')
plt.colorbar()
plt.show()


#####################################################################################################################
#####                                         LOAD PRETRAINED EMBEDDING                                         #####
#####################################################################################################################

# Load pre-trained GloVe100 word embeddings
# Create the dictionnary to gather the word and its embedding (word as key and embedding as value)
embeddings_dict = {}
# Define the directory where the GloVe file is
GLOVE_DIR = "glove/"
# Open the GloVe file
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf8")
# Iterate on the line
for line in f:
    # Split the line
    values = line.split()
    # Get the 1st element, which is the word
    word = values[0]
    # Get the word embedding, which is the rest of the line
    coefs = np.asarray(values[1:], dtype='float32')
    # Add the word as a new key and the coefs as its value
    embeddings_dict[word] = coefs
f.close()

# Print the number of words and the word embedding dimension 
print('Found',len(embeddings_dict),'word vectors.')
print('Dimension of 1 word embedding is:', embeddings_dict['hi'].shape)

# Create 2 sentences with same words but different order 
# (sentence 1 groups the words with similar meaning while sentence 2 orders them randomly)
texts = ['king queen man woman dog wolf football basketball red green yellow',
         'man queen yellow basketball green dog  woman football  king red wolf']

# Apply the tokenization to the raw text
# Create an instance of the Tokenizer class from Keras. Argument num_words limits the tokenizer to only the top MAX_NB_WORDS most frequent words in the corpus
tokenizer = Tokenizer(num_words = MAX_NB_WORDS+1)
# Train the tokenizer on texts and create a word index (where each word is assigned to a unique integer index based on its frequency in the dataset)
tokenizer.fit_on_texts(texts)
# Convert each text into a sequence of integers, where each integer represents the index of a word in the word index learned by the tokenizer
sequences = tokenizer.texts_to_sequences(texts)

# Get the dictionnary mapping each word in texts to an index
word_index = tokenizer.word_index
print('Found',len(word_index),'unique tokens')

# Pad each sequence to the MAX_SEQUENCE_LENGTH, so add 0 at the end of each sequence such as we get MAX_SEQUENCE_LENGTH tokens per sequence
data = pad_sequences(sequences, padding='post', maxlen=MAX_SEQUENCE_LENGTH)

# 2 sentences with 50 values (MAX_SEQUENCE_LENGTH)
print(data.shape)

# Each word was replaced by the index matching it
print(data)


# Get the embeddings for the different words that appear in the text
# Create a matrix full of 0. We take len(word_index) + 1 rows to keep one row full of 0 for padding or unknown tokens
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
# Iterate on the words
for word, i in word_index.items():
    # Get the embedding of the specific word
    embedding_vector = embeddings_dict.get(word)
    # Words not found in embedding index will be left to all-zeros
    if embedding_vector is not None:
        # Add the embedding to the matrix if it exists
        embedding_matrix[i] = embedding_vector
print(embedding_matrix.shape)


# Create an embedding layer using the weights extracted from the pre-trained GloVe100 embeddings
embedding_layer = Embedding(
    # Size of the vocabulary
    len(word_index) + 1,
    # Dimension of the dense embedding
    EMBEDDING_DIM,
    # Initializer for the embeddings matrix
    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
    # Whether to train, here we don't because our sample is too small (and we don't run a model)
    trainable=False)


# Transform the input tokenized data to the embedding using the Embedding layer
embedding = embedding_layer(data)

# Check the shape of the embedding (the last dimension of this matrix contains the embeddings of the words in the sentence)
print(embedding.shape)


#####################################################################################################################
#####                                     VISUALIZATION OF WORD EMBEDDINGS                                      #####
#####################################################################################################################

# Create the PCA function (to reduce the dimension of embedding from 100 to 2)
def pca(embedding, sequences, sentence):
    '''
    Use PCA to reduce the dimension of the embedding (to plot it on a 2D graph)

    Inputs:
    embedding -- embedding matrix
    sequences -- tokenized version of the raw input (list of indexes)
    sentence -- index of the sentence/raw input

    Returns:
    pca_array -- array of dimension (len(sequences[sentence]),n_components) extracting the main information from the embedding matrix
    '''
    # Set up a PCA instance keeping 2 dimensions
    pca = PCA(n_components=2)

    # Fit the PCA instance to our data (using the specific sentence and all its elements)
    pca_array = pca.fit_transform(embedding[sentence, 0:len(sequences[sentence]), :])

    return pca_array


# Create the plot function (on a Cartesian plane)
def pca_plot(sequences, sentence, pca_array):
    '''
    Plot the PCA values on a Cartesian plane

    Inputs:
    sequences -- tokenized version of the raw input (list of indexes)
    sentence -- index of the sentence/raw input
    pca_array -- array of dimension (len(sequences[sentence]),n_components) extracting the main information from the embedding matrix
    '''
    # Set up the font size
    plt.rcParams['font.size'] = '12'
    
    # Plot using the values we got from PCA (use PC1 as x index and PC2 as y index)
    plt.scatter(pca_array[:, 0], pca_array[:, 1])

    # Annotate the word to the right point on the graph
    words = list(word_index.keys())
    for i, index in enumerate(sequences[sentence]):
        # We use index-1 because sequences values start with 1 while words index start with 0
        plt.annotate(words[index-1], (pca_array[i, 0], pca_array[i, 1]))


# Compare embeddings for each sentence (the order of the words does not affect the vector representation)
plt.subplot(1,2,1)
pca_plot(sequences, 0, pca(embedding, sequences, 0))
plt.title('Sentence: '+texts[0], fontsize=8)
plt.subplot(1,2,2)
pca_plot(sequences, 1, pca(embedding, sequences, 1))
plt.title('Sentence: '+texts[1], fontsize=8)
plt.suptitle('Plot the word embeddings of each sentence')
plt.show()


#####################################################################################################################
#####                           VISUALIZATION OF WORD EMBEDDING & POSITIONAL ENCODING                           #####
#####################################################################################################################

# Determine equal weights for word embedding and positional encoding
embedding_weight = 1
positional_weight = 1

# Get the weighted sum of word embedding and positional encoding
embedding_pos_encoding = embedding * embedding_weight + pos_encoding[:,:,:] * positional_weight

# Plot the relationship between word embedding and positional encoding for each sentence to compare (first apply PCA)
plt.subplot(1,2,1)
pca_plot(sequences, 0, pca(embedding_pos_encoding, sequences, 0))
plt.title('Sentence: '+texts[0], fontsize=8)
plt.subplot(1,2,2)
pca_plot(sequences, 1, pca(embedding_pos_encoding, sequences, 1))
plt.title('Sentence: '+texts[1], fontsize=8)
plt.suptitle('Plot the word embedding (weight='+str(embedding_weight)+
             ') combined with positional encoding (weight='+str(positional_weight)+') of each sentence')
plt.show()


# Determine higher weights for word embedding than positional encoding
embedding_weight = 10
positional_weight = 1

# Get the weighted sum of word embedding and positional encoding
embedding_pos_encoding = embedding * embedding_weight + pos_encoding[:,:,:] * positional_weight

# Plot the relationship between word embedding and positional encoding for each sentence to compare (first apply PCA)
plt.subplot(1,2,1)
pca_plot(sequences, 0, pca(embedding_pos_encoding, sequences, 0))
plt.title('Sentence: '+texts[0], fontsize=8)
plt.subplot(1,2,2)
pca_plot(sequences, 1, pca(embedding_pos_encoding, sequences, 1))
plt.title('Sentence: '+texts[1], fontsize=8)
plt.suptitle('Plot the word embedding (weight='+str(embedding_weight)+
             ') combined with positional encoding (weight='+str(positional_weight)+') of each sentence')
plt.show()


# Determine lower weights for word embeddings than positional encoding
embedding_weight = 1
positional_weight = 10

# Get the weighted sum of word embedding and positional encoding
embedding_pos_encoding = embedding * embedding_weight + pos_encoding[:,:,:] * positional_weight

# Plot the relationship between word embedding and positional encoding for each sentence to compare (first apply PCA)
plt.subplot(1,2,1)
pca_plot(sequences, 0, pca(embedding_pos_encoding, sequences, 0))
plt.title('Sentence: '+texts[0], fontsize=8)
plt.subplot(1,2,2)
pca_plot(sequences, 1, pca(embedding_pos_encoding, sequences, 1))
plt.title('Sentence: '+texts[1], fontsize=8)
plt.suptitle('Plot the word embedding (weight='+str(embedding_weight)+
             ') combined with positional encoding (weight='+str(positional_weight)+') of each sentence')
plt.show()


