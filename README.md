# pyLDA
Topic Modeling with latent dirichlet allocation (LDA):
This code performs topic modeling on one directory (i.e., /4) of the corpus of conversations provided at: http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/. It uses scikit's LDA implementation of online learning of LDA [1].

# Requirements:
To run this code, you need the following:
1) python3
2) nltk
3) numpy
4) sklearn


# Preprocessing:
The following preprocessing steps are performed:
1) tokenization and lemmatization with nltk
2) removal of stop words, urls, and puctuations
3) lowercasing, removal of highly frequent, and removal of rare words


# Running:
You can perform LDA on the corpus and predict the topic distribution of a test file:

python3 lda.py --trainpath path-to-corpus --testfile path-to-testfile
 
By default, the model trains on a subset of the dataset (20k convesations). To train on the full corpus:

python3 lda.py --trainpath path-to-corpus --testfile path-to-testfile  --max -1


# Hyperparameters:
The hyperparameters are found by a grid search on the number of topics {40, 50} and learning decay rate {0.7, 0.9}
The best model has the following properties:
Log likelihood score:  -8156299.135630326
Perplexity:  5884.502890047591

top topics in the corpus
TOPIC: aptitude command actually synaptic router getting package update aptget sudo
TOPIC: machine pc want guy hey live time cd linux window
TOPIC: called deb bad connection dvd using having network problem driver
TOPIC: setup anybody hello answer want hi desktop ask good question
TOPIC: script gui manager application add download line make command run

# References:
[1] Hoffman, Matthew, Francis R. Bach, and David M. Blei. "Online learning for latent dirichlet allocation." NIPS. 2010.
