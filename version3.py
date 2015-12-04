# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:43:48 2015

@author: Amulya
"""

"""
Current Version:
    Use naive_bayes.MultinomialNB;
    Exclude stopwords;

Update Log:
V1.10   Use naive_bayes.MultinomialNB; Exclude stopwords;
        
V1.03   Use Linear SVC; Exclude stopwords;
        0.71714646
V1.02:  Use SVC; Exclude stopwords;
        Failed
V1.01:  Use 3 Neighbors; Exclude stopwords;
        0.69962454
V1.00:  Use logistic regression; Exclude stopwords;
        0.7571965
"""


import re
#from nltk import pos_tag
from nltk.corpus import stopwords
from string import digits
from sklearn.feature_extraction.text import CountVectorizer
#from nltk.classify import SklearnClassifier
#import nltk
#import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from nltk.util import ngrams
#from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


#read the reviews and their polarities from a given file
def loadData(fname):
    reviews=[]
    labels=[]
    f=open(fname)
    for line in f:
        review,rating=line.strip().split('\t')  
        reviews.append(review.lower())    
        labels.append(int(rating))
    f.close()
    return reviews,labels
    
def loadtestData(fname):
    datas=[]
    f=open(fname)
    for data in f:
        datas.append(data.lower().strip())    
    f.close()
    return datas

rev_train,labels_train=loadData('training.txt')
#rev_test=loadtestData('testing.txt')



'''
rev_test=rev_train[2001:2500]
rev_train=rev_train[0:2000]
labels_test=labels_train[2001:2500]
labels_train=labels_train[0:2000]
'''

rev_test=rev_train[:2000]
rev_train=rev_train[2001:]
labels_test=labels_train[0:2000]
labels_train=labels_train[2001:]



###Process data
def ProcessData(train_text):
    
    #adjectives = set()
    """
    for sentence in sentences_set[:3]:
        
        #sentence=re.sub('n\'t',' not',sentence)#replace abbreviation
        sentence=re.sub('[^a-z\d]',' ',sentence)#replace chars that are not letters or numbers with a space
        sentence=re.sub(' +',' ',sentence)#remove duplicate spaces
        sentence= re.sub(r"http\S+", "", sentence) ## remove hyperlinks
        sentence= re.sub(r'^https?:\/\/.*[\r\n]*', '', sentence) # remove hyperlinks
        sentence= sentence.encode('utf8') # decoding data
    """
    for i in range(len(train_text)):
        #train_text[i]=strip_digits(train_text[i]) # Stripping integer
        train_text[i]=re.sub("\d+", "", train_text[i])
        train_text[i] = re.sub(r"http\S+", "", train_text[i]) #re.sub(r'^http?:\/\/.*[\r\n]*', '', train_text[i]) # remove hyperlinks
        train_text[i] = re.sub(r'[?|$|.|!]',r'',train_text[i]) # removing punctuation    
        
        train_text[i]=train_text[i].encode('utf8')  # decoding data  
        train_text[i]=train_text[i].lower() #lower
        train_text[i]=re.sub('[^a-z\d]',' ',train_text[i])
        train_text[i]=re.sub(' +',' ',train_text[i]) # removing duplicate spaces

    return train_text

def CombineNot(sentences):  
    new_sentences_set=[]
        #tokenize the sentence 
    for sentence in sentences:       
        terms = sentence.split()
        twograms = ngrams(terms,2) #compute 2-grams
      
        for tg in twograms:  
        #print tg
            if (tg[0].lower() == 'not'): 
                a = ''.join(tg)
                #print a    
                sentence = sentence.replace(tg[0]+ ' ' + tg[1], a)
        #print sentence
        new_sentences_set.append(' '.join(sentence))
        
       
    return new_sentences_set  
    

rev_train=ProcessData(rev_train)
rev_train=CombineNot(rev_train)
rev_test=ProcessData(rev_test)
rev_test=CombineNot(rev_test)



###Analysis Data
#Build a counter based on the training dataset
counter = CountVectorizer(analyzer = 'word', stop_words=set(stopwords.words('english')),ngram_range=(1,2))
counter.fit(rev_train) #only focus on the text appear in rev_train
   
#count the number of times each term appears in a document and transform each doc into a count vector
counts_train = counter.transform(rev_train)#transform the training data
counts_test = counter.transform(rev_test)#transform the testing data

clf1 = LogisticRegression()
clf2 = KNeighborsClassifier(3)
clf3 = MultinomialNB()
clf4 = DecisionTreeClassifier(max_depth=20, max_features=None, class_weight=None)
#clf5 = svm.SVC()

 
#build a fit on the training data
#classifier = MultinomialNB()
#classifier = RandomForestClassifier(n_estimators = 50) 
#classifier = DecisionTreeClassifier(max_depth=20, max_features=None, class_weight=None)
classifier = VotingClassifier(estimators=[('dt', clf4), ('lr', clf1), ('mnb', clf3)], voting='soft', weights=[2, 3, 3])

classifier.fit(counts_train, labels_train)

#use the classifier to predict
predicted=classifier.predict(counts_test)

resultwriter=open('out.txt','w')
for each_predicted in predicted:
    resultwriter.write(str(each_predicted)+'\n')
resultwriter.close()



#'''
###Score
def writein(vari,file_name):
    resultwriter=open(file_name,'w')
    for each in vari:
        resultwriter.write(str(each)+'\n')
    resultwriter.close()

writein(rev_train,'adj-train.txt')
writein(rev_test,'adj-test.txt')

from sklearn.metrics import accuracy_score 

print (accuracy_score(predicted,labels_test))
#'''