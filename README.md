# Positional encoding

## Project description
In this project, we are going to study some pre-processing methods we apply to raw text before passing it to the encoder and decoder blocks of the Transformer architecture. We are going to create visualizations to gain intuition on positional encodings and to understand how positional encodings affect word embeddings.


## Positional encodings
In NLP, we usually convert sentences into tokens before feeding texts into a language model. Each token is then converted into a numerical vector of fixed length called an embedding, which captures the meaning of the word.
In the Transformer architecture, we add a positional encoding vector to the embedding to pass positional information (order of tokens in a sequence) throughout the model. Indeed with Transformer architecture, the whole data is passed at once, so, (without positional encoding) we would lose the information about the order of the data.

With positional encoding vectors, we notice that:
* words that are closer in a sentence appear closer when plotted on a Cartesian plane
* words that are further in a sentence appear further on the plane

Positional encoding is computed using sine and cosine functions enabling the model to capture sequential information while processing the sequence in parallel. This method allows the Transformer to perform well on tasks that require an understanding of token position, such as translation or language modeling.

Properties of the positional encoding matrix:
* the norm of each of the vectors is always a constant
* the norm of the difference between 2 vectors separated by k positions is also constant. This property is important because it demonstrates that the difference does not depend on the positions of each encoding, but rather the relative seperation between encodings. Being able to express positional encodings as linear functions of one another can help the model to learn by focusing on the relative positions of words.

The positional encoding matrix help to visualize how each vector is unique for every position. 


### Visualization of positional encoding with correlation
We calculate the correlation between pairs of vectors at every single position. A successful positional encoder will produce a perfectly symmetric matrix:
* maximum values are located at the main diagonal: vectors in similar positions should have the highest correlation
* smaller values are located away from the main diagonal


### Visualization of positional encoding with Euclidean distance
The visualization will display a matrix in which:
* the main diagonal is 0 (because the distance between same vector is null)
* its off-diagonal values increase as they move away from the main diagonal


## Dataset 
We use the pre-trained GloVe100 word embeddings. It has 400k words and each word embedding has 100 features.

We use 2 sentences:
* each sentence has same words (11) but ordered differently
* each words is part of a group of 2 or 3 words with similar meaning
* 1 sentence has the words ordered by meaning, and the other one has words randomly ordered


### Data preprocessing
We tokenize and pad the raw text. We get a matrix with one row for each sentence, each of them represented by an array of size MAX_SEQUENCE_LENGTH. Each value in this array is the index mapped from the inital word/token. The sequences shorter than MAX_SEQUENCE_LENGTH are padded with zeros to create uniform length. 


## Word embeddings
The embedding array has dimensions too hard to plot on a 2D graph. So, to make the visualization easier, we use PCA to reduce the dimensions (from 100 features of the GloVe embedding to only 2 components). 
The plot is the same for both sentence, meaning that if we take only the embedding, the data is the same so the order information is not kept.

