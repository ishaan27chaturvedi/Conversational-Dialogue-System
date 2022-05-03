# Conversational-Dialogue-System

Creating a Conversational Dialogue System with Dialogue Act Tagging


## Description

### Dialogue Act tagging
Dialogue act (DA) tagging is an essential step in the development of dialog systems. DA tagging is a problem that is usually solved using supervised machine learning techniques, which all require a large amount of manually labeled data. For DA tagging, a variety of methods have been investigated. We look at two different DA classification models. The [Switchboard Dialog Act Corpus](http://compprag.christopherpotts.net/swda.html) is being used for training. This dataset contains 43 tags. Yes-No-Question ('qy'), Statement-non-opinion ('sd'), and Statement-opinion ('sv') are some of the tags. Tags information can be found [here](http://compprag.christopherpotts.net/swda.html#tags). 
<br>
For **model 1**, We create a sequential model with an embedding layer, with 2 Bidirectional LSTMs with size 43 (number of unique tags), followed by a dense layer and a softmax activation layer as output. Resulting Overall Accuracy: 68.76497864723206
<br>
As this dataset has a highly imbalanced, we use compute_class_weights from sklearn to create more balanced weights for **model 2**. Here we use the same model as before but we instead pass it class weights to balance out the under represented classes. Since we assigned more weights to the minority classes, the model was able to predict more properly for these classes. We see that the accuracy for bf and br has increased slightly but the overall accuracy has decreased significantly.
<br>
For **model 3**, we concat the maxpool tensors, use TimeDistributed layer to have the CNN layer interact with LSTM, flatten and join it with the concat maxpool tensors. We join this with the embedding layer and add a dropout. Then we create 2 bidirectional LSTM, a dense layer with softmax activation, flatten it and add a dropout. Finally, we concat the flattened layer with the dropout and pass it to the output layer with a softmax activation
<br><br>
From the above 3 models we can see that adding context did help in better prediction of minority classes.

### Conversational Dialogue Systems
We create end-to-end dialogue systems using seq2seq Machine Translation. Customer support apps and online helpdesks are among the places where conversational models can be used. Retrieval-based models, which produce predefined responses to questions of specific types, are often used to power these models. But here, we use seq2seq model to build a generative model.
<br>
We use the [Cornell Movie Dialogues Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) that contains 220,579 conversational exchanges between 10,292 pairs of movie characters, 9,035 characters from 617 movies, and 304,713 total utterances. This dataset is large with a wide variety of language formality, time periods, and other variables. Our hope is that this variety will make our model responsive to a wide range of queries.
<br>
We use pretrained Glove Embeddings for our embed layer. A sequence-to-sequence (seq2seq) model is at the core of our model. The purpose of a seq2seq model is to use a fixed-sized sequence as an input and generate a variable-length sequence as an output.

#### Encoder
The encoder RNN iterates through the input sentence one token at a time, producing an "output" vector and a "hidden state" vector at each time step. The output vector is recorded while the hidden state vector is transferred to the next time step. The encoder converts the context it observed at each point in the sequence into a set of points in a high-dimensional space, which the decoder can use to produce a meaningful output for the task at hand.<br>
A multi-layered Gated Recurrent Unit, created by Cho et al., is at the centre of our encoder. We'll use a bidirectional version of the GRU, which effectively means there are two separate RNNs: one fed the input sequence in regular sequential order and the other fed the input sequence in reverse order. At each time point, the outputs of each network are added together.

#### Decoder
The response utterance is produced token by token by the decoder RNN. It generates the next word in the sequence using the encoder's context vectors and internal hidden states. It keeps producing words until it reaches the end of the sentence, which is represented by an end_token. A common issue with a standard seq2seq decoder is that relying solely on the context vector to encode the meaning of the complete input sequence would almost certainly result in information loss. This is particularly true when dealing with long input sequences, severely restricting our decoder's capabilities.<br>
Bahdanau et al. devised an "attention mechanism" that allows the decoder to focus on specific parts of the input sequence rather than using the whole set context at each step to deal with information loss. Attention is determined using the encoder's outputs and the decoder's current hidden state. Since the output attention weights have the same shape as the input sequence, we may multiply them by the encoder outputs to get a weighted amount that shows which sections of the encoder output to focus on.


## Getting Started

### Dependencies

* Python
* NLTK
* Tensorflow
* Keras
* Transformers (huggingface)
* sklearn
* Pandas
* Matplotlib
* Numpy
* Requests
* tqdm

### Installing

* Download the notebook 
* Download the dataset for the respective notebook:
  * Dialogue Act Tagging - [Switchboard Dialog Act Corpus](http://compprag.christopherpotts.net/swda.html)
  * Cobversational Dialogue Systems - [Cornell Movie Dialogues Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
* Place the dataset in the same directory as the notebook

### Executing program

* Run the notebook in your preferref environment (Colab or Jupyter)


## Help

If you've found a new bug, go ahead and create a new GitHub issue. Be sure to include as much information as possible so I can reproduce the bug.


## Authors

Prof Massimo Poesio 
<br>Professor of Computational Linguistics 
<br>Queen Mary University of London
