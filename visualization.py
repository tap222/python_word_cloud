import nltk
import traceback
import collections
import numpy as np
import pandas as pd 
import networkx as nx
from nltk import bigrams
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


import warnings
warnings.filterwarnings("ignore")

def plotmostfrqKwds(pData, n, filename):
    try:
        pData = pData.reset_index()
        pData = pData[~pData.Sample.str.contains("nan")]
        pDescWords = '' 
        #iterate through the csv file 
        for val in pData.Sample: 
            #typecaste each val to string 
            val = str(val) 
            # split the value 
            tokens = val.split()        
            pDescWords += " ".join(tokens)+" "
  
        vect = TfidfVectorizer()
        X = vect.fit_transform([pDescWords])
        #zipping actual words and sum of their Tfidf for corpus
        features_rank = list(zip(vect.get_feature_names(), [x[0] for x in X.sum(axis=0).T.tolist()]))

        # sorting
        features_rank = np.array(sorted(features_rank, key = lambda x:x[1], reverse=True))
        plt.figure(figsize = (12,14))
        plt.barh(-np.arange(int(n)), features_rank[:int(n), 1].astype(float), height=.8)
        plt.yticks(ticks=-np.arange(int(n)), labels = features_rank[:int(n), 0])
        plt.title("Most Frequent Keywords",fontsize = 25)       
        plt.savefig('./output/FrequentKeywords' + str(filename) + '.png')
        
    except Exception as e:
        print('Error[001] ocurred visualization file')
        print(traceback.format_exc())
        return(-1)
    return (0)
    
def plottablefrqKwds(pData, n, filename):
    try:
        pData = pData.reset_index()
        pData = pData[~pData.Sample.str.contains("nan")]
        pDescWords = pData.Sample.str.cat(sep=' ')
        pWords = nltk.tokenize.word_tokenize(pDescWords)
        pWordDist = nltk.FreqDist(pWords)
        pFrqKwdsDf = pd.DataFrame(pWordDist.most_common(n), columns=['TopWord', 'Frequency'])
        pFrqKwdsDf.to_excel('./output/FrqKwds' + str(filename) + '.xlsx', index = False)
        
    except Exception as e:
        print('Error[002] ocurred creating  table  for top n keywords')
        print(traceback.format_exc())
        return(-1)
    return(0)       
    
def get_top_ngram(corpus, n, ngram):
    try:
        vec = CountVectorizer(ngram_range = (int(ngram), int(ngram)) ).fit(corpus)
        pBagOfWords = vec.transform(corpus)
        pSumWords = pBagOfWords.sum(axis=0) 
        pWordsFreq = [(word, pSumWords[0, idx]) for word, idx in vec.vocabulary_.items()]
        pWordsFreq =sorted(pWordsFreq, key = lambda x: x[1], reverse=True)
        
    except Exception as e:
        print('Error[003] ocurred visualization file while getting top n grams')
        print(traceback.format_exc())
        return(-1)  
    
    return pWordsFreq[:int(n)]

def plotFreqngram(pData, n, ngram, filename):
    try:
        pData = pData.reset_index()
        pData = pData[~pData.Sample.str.contains("nan")]
        pDescWords = '' 
        #iterate through the csv file 
        for val in pData.Sample:  
            #typecaste each val to string 
            val = str(val) 
            # split the value 
            tokens = val.split() 
            pDescWords += " ".join(tokens)+" "
            
        common_words = get_top_ngram([pDescWords], n, ngram)
        fig, ax = plt.subplots(figsize=(20, 20))
        pNgramDf = pd.DataFrame(common_words, columns = ['NgramKeyword' , 'count'])
        pNgramDf.to_excel('./output/Ngrarm' + str(filename) + '.xlsx', index = False)
        pNgramDf.groupby('NgramKeyword').sum()['count'].sort_values(ascending=False).plot.barh(x='NgramKeyword', y='count',ax=ax, color="purple")
        plt.title("Ngram Keywords",fontsize = 25)       
        plt.savefig('./output/NgramFreq' + str(filename) + '.png')
        
    except Exception as e:
        print('Error[003] ocurred visualization file')
        print(traceback.format_exc())
        return(-1)
    return (0)
    
def plotngramnetwork(pData, pNodeName, filename):
    try:
        pData = pData.reset_index()
        pData = pData[~pData['Sample'].str.contains("nan",na=False)]
        pDescWords = '' 
        #iterate through the csv file 
        for val in pData['Sample']: 
              
            #typecaste each val to string 
            val = str(val) 
          
            # split the value 
            tokens = val.split() 
                    
            pDescWords += " ".join(tokens)+" "
            
        terms_bigram = list(bigrams(pDescWords.split()))
        
        # Create counter of words in clean bigrams
        bigram_counts = collections.Counter(terms_bigram)
        
        bigram_df = pd.DataFrame(bigram_counts.most_common(int(20)), columns=['ngram', 'count'])
        # Create dictionary of bigrams and their counts
        d = bigram_df.set_index('ngram').T.to_dict('records')
        
        # Create network plot 
        G = nx.Graph()

        # Create connections between nodes
        for k, v in d[0].items():
            G.add_edge(k[0], k[1], weight=(v * 10))

        G.add_node(str(pNodeName), weight=100)
            
        fig, ax = plt.subplots(figsize=(12,14))

        pos = nx.spring_layout(G, k=2)

        # Plot networks
        nx.draw_networkx(G, pos,
                         font_size=16,
                         width=3,
                         edge_color='grey',
                         node_color='purple',
                         with_labels = False,
                         ax=ax)

        # Create offset labels
        for key, value in pos.items():
            x, y = value[0]+.135, value[1]+.045
            ax.text(x, y,
                    s=key,
                    bbox=dict(facecolor='red', alpha=0.25),
                    horizontalalignment='center', fontsize=13)
        plt.title("Ngram Keywords",fontsize = 25)       
        plt.savefig('./output/Ngramnetworks' + str(filename) + '.png')
        
    except Exception as e:
        print('Error[003] ocurred visualization file')
        print(traceback.format_exc())
        return(-1)
    return (0)