#Author: Javid Ebrahimi  <javid@cs.uoreogn.edu>
import os
import sys
from time import time
import re
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import PredefinedSplit, GridSearchCV
import argparse
import numpy as np

'''
This tokenizer is taken from http://scikit-learn.org/stable/modules/feature_extraction.html
'''
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def strip_str(txt):
    #remove URLs
    txt = re.sub(r'^https?:\/\/.*[\r\n]*', '', txt, flags=re.MULTILINE)
    #remove punctuation 
    txt = re.sub(r"([^\s\w]|_)+", '', txt)
    return txt
    
def print_topics(doc_topic):
    for topic in np.argsort(doc_topic.sum(axis=0))[-args.toptopics:]:
        words = 'TOPIC: ' + ' '.join([features[i]
                   for i in lda.components_[topic].argsort()[-args.topwords:]])
        print(words)

def train():
    assert(os.path.exists(args.trainpath)), "path to training files does not exist"
    files = os.listdir(args.trainpath)
    assert(len(files) > 0), "empty directory"
    corpus = []
    for file_ in files:
        with open(args.trainpath+file_, encoding="utf8", errors='ignore') as f:
            doc = []
            for line in f:
                line = line.strip()
                if line:
                    doc.append(strip_str(line.split('\t')[-1]))
        corpus.append(' '.join(doc))   
    X = vectorizer.fit_transform(corpus)    
    X = shuffle(X, random_state=args.seed)
    if args.max != -1:
       X = X[:args.max,:]
    #perform grid search and replace the default model with the best model on validation set
    #lda = grid_search(X)
    #Document representation based on the topic distribution
    docs_topic = lda.fit_transform(X)
    print('Perplexity: ', lda.perplexity(X)) 
    return docs_topic, lda    

def test():
   assert(os.path.exists(args.testfile)), "path to test file does not exist"
   with open(args.testfile, encoding="utf8", errors='ignore') as f:
        doc = []
        for line in f:
            line = line.strip()
            if line:
               doc.append(strip_str(line.split('\t')[-1]))
   doc = vectorizer.transform([' '.join(doc)])
   if doc.count_nonzero() != 0:
      topic_distribution = lda.transform(doc)  
      print('Top topics the test instance belongs to')
      print_topics(topic_distribution)
   else:
      print('Not enough words to classify topic', file=sys.stderr)

'''
The code to do a grid search on the params of LDA, based on the log likelihood score on validation set 
'''
def grid_search(X):
    num_training_total = X.shape[0]
    num_val = num_training_total // 5    
    val_fold = (num_training_total - num_val)*[-1] + num_val*[0]
    ps = PredefinedSplit(val_fold)
    lda = LatentDirichletAllocation(learning_method='online', random_state=args.seed, max_iter=10)
    search_params = {'n_components': [40, 50], 'learning_decay': [.7, .9]} 
    grid_search = GridSearchCV(lda, param_grid=search_params, cv=ps)
    grid_search.fit(X)
    best_model = grid_search.best_estimator_
    print('Best params: ', grid_search.best_params_)
    print('Log likelihood score: ', grid_search.best_score_)
    return best_model 

def parse_args():
    parser = argparse.ArgumentParser(
             description=__doc__,
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--trainpath', help="Path to conversation files for training", default="dialogs/4/", type=str)
    parser.add_argument('--testfile', help="Path to test file", default="dialogs/4/221499.tsv", type=str)
    parser.add_argument('--topics', help="Number of topics", default=40, type=int)
    parser.add_argument('--toptopics', help="number of top topics to show", default=5, type=int)
    parser.add_argument('--topwords', help="number of top words in each topic to show", default=10, type=int)
    parser.add_argument('--seed', help="Random seed", default=123, type=int)
    parser.add_argument('--max',help="Train on a smaller subset, set by the max number of conversations."
                                      "Set to -1 for full training", default=20000, type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words='english', min_df=5, max_df=0.1) 
    lda = LatentDirichletAllocation(learning_method='online', random_state=args.seed, 
                                    n_components=args.topics, learning_decay=.9, max_iter=10)
    print('training...')
    t = time()
    docs_topic, lda = train()
    print('finished in %0.4fs.' % (time() - t))
    features = vectorizer.get_feature_names()
    print('top topics in the corpus')
    print_topics(docs_topic)
    print('testing')
    test()

