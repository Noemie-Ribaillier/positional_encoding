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

