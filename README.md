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
* the norm of the difference between 2 vectors separated by k positions is also constant. This property is important because it demonstrates that the difference does not depend on the positions of each encoding, but rather the relative seperation between encodings. Being able to express positional encodings as linear function of one another can help the model to learn by focusing on the relative position of words.

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
We load the GloVe100 word embedding and we keep the embeddings from the words that match the words in our sentence. So, we end up with a smaller embedding matrix (number of words + 1 to keep a case of unkown word).
We create an embedding layer and we apply it on our tokenized data. We end up with a matrix of shape (number of rows in the text, number of max words in the sequence, number of items in the embedding). So we get the word embedding of each word in each sentence, ordered by the word positions.

This embedding array has dimensions too hard to plot on a 2D graph. So, to make the visualization easier, we use PCA to reduce the dimensions (from 100 features of the GloVe embedding to only 2 components). 
The plot is the same for both sentence, meaning that if we take only the embedding (even though it's ordered per words in the sentence), the data is the same so the order information is not kept.


### PCA
Principal Component Analysis (PCA) is a dimensionality reduction technique used to reduce the number of features (dimensions) in a dataset while retaining as much of the original variance (information) as possible. It transforms the original features into a new set of orthogonal (uncorrelated) features called principal components.

Main steps of the PCA:
* Standardize the data: PCA works best when the data is normalized (mean = 0 and variance = 1) for each feature
* Compute the covariance matrix: the covariance matrix captures the relationships between features in the data (how features vary together)
* Calculate eigenvalues and eigenvectors: these represent the directions (principal components) in which the data has the most variance. The eigenvalues tell us how much variance there is in the direction of the corresponding eigenvector
* Sort the eigenvalues and eigenvectors: the eigenvectors are sorted in descending order of their eigenvalues. The first eigenvector corresponds to the direction of the highest variance in the data
* Select the top k components: choose the top k eigenvectors that correspond to the largest eigenvalues, which represent the most important directions in the data. In our case, because we want to plot on a 2D graph, we choose k=2.
* Transform the data: project the original data onto the selected principal components to reduce the dimensionality


## Word embedding vs positional encoding
In this part, we compare if the positional encoding influence the word embeddings.
Then, we are setting up different weights for embeddings and positional encodings to check their relationship.


### Relationship between word embeddings and positional encoding (when weights are equal)
We notice that the plot changed compared to when we only take into account the word embeddings. In the plot which corresponds to the sentence where words are randomly ordered, some words appear more close by such as "red" and "wolf" (because they are close by in the sentence).


### Relationship between word embeddings and positional encoding (when embeddings weight are higher than positional encoding weight)
The plot looks very similar to the original embeddings visualization and there are only a few changes between the positions of the plotted words. The word embeddings are dominating the positional encoding vectors.


### Relationship between word embeddings and positional encoding (when embeddings weight are lower than positional encoding weight)
The arrangement of the words takes a clockwise order. The positional encoding vectors are dominating the embedding. 


## References
This script is coming from the Deep Learning Specialization course. I enriched it to this new version.
