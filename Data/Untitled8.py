
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import BernoulliNB
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn import cross_validation


# In[2]:


df=pd.read_csv('AllProductReviews.csv')


# In[3]:


df.head()


# In[4]:


def partition(x):
    if x <= 3:
        return 0
    return 1

#changing reviews with score less than 3 to be positive and vice-versa
actualScore = df['ReviewStar']
positiveNegative = actualScore.map(partition) 
df['ReviewStar'] = positiveNegative


# In[5]:


df.shape


# In[6]:


final=df.drop_duplicates(subset={"Product","ReviewBody","ReviewTitle"}, keep='first', inplace=False)
final.shape


# In[16]:


final_data=final[0:1000]


# In[17]:


def textCleaning(df):
    tokens = []
    df=pd.DataFrame(df)
    for i in range(len(df)):
        tokens.append(word_tokenize(df['ReviewBody'].iloc[i]))

    stopwordsList = stopwords.words("english")
    stopwordsList.extend([',','.','-','!'])

    wordsList = []
    for tokenList in tokens:
        words = []
        for word in tokenList:
            if word.lower() not in stopwordsList:
                words.append(word.lower())
        wordsList.append(words)

    wnet = WordNetLemmatizer()
    for i in range(len(wordsList)):
        for j in range(len(wordsList[i])):
            wordsList[i][j] = wnet.lemmatize(wordsList[i][j], pos='v')

    for i in range(len(wordsList)):
        wordsList[i] = ' '.join(wordsList[i])

    return wordsList


# In[18]:



from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
wordlist=textCleaning(final_data)


# In[19]:


count_vect = CountVectorizer(ngram_range=(1,2)) #in scikit-learn
final_counts = count_vect.fit_transform(final_data['ReviewBody'].values)
final_counts=final_counts.toarray()


# In[20]:


y = np.array(final_data['ReviewStar'])

from sklearn.preprocessing import StandardScaler
standardised_data=StandardScaler(with_mean= False ).fit_transform(final_counts)


# In[21]:


X_train, X_test, y_train, y_test = cross_validation.train_test_split(standardised_data,y,test_size=0.2, random_state=0)
myList = list(range(0,50))
neighbors = list(filter(lambda x: x % 2 != 0, myList))
cv_scores = []
# perform 10-fold cross validation
for k in neighbors:
    clf = BernoulliNB(alpha=k, binarize=0.0, fit_prior=True, class_prior=None)
    scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_alpha = neighbors[MSE.index(min(MSE))]
print('\nThe optimal alpha is %d.' % optimal_alpha)

# plot misclassification error vs k 
plt.plot(neighbors, MSE)

for xy in zip(neighbors, np.round(MSE,3)):
    plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')

plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()

print("the misclassification error for each k value is : ", np.round(MSE,3))


# In[31]:


clf_optimal = BernoulliNB(alpha=1, binarize=0.0 , fit_prior=False, class_prior=[1,1])


# In[32]:


# fitting the model
clf_optimal.fit(X_train, y_train)


# In[33]:


pred = clf_optimal.predict(X_test)

# evaluate accuracy
acc = accuracy_score(y_test, pred) * 100
print(acc)


# In[34]:


from sklearn.metrics import confusion_matrix
clf_optimal = BernoulliNB(alpha=optimal_alpha , binarize=0.0 , fit_prior=False, class_prior=[1,1])

# fitting the model
clf_optimal.fit(X_train, y_train)
print(clf_optimal.coef_)
# predict the response
pred = clf_optimal.predict(X_test)

# evaluate accuracy
matrix = confusion_matrix(y_test, pred) 
tn,fp,fn,tp= confusion_matrix(y_test, pred).ravel()
print(tn,fp,fn,tp)
precision=tp/(tp+fp)
recall=tp/(fn+tp)
f1=(2*((precision*recall)/(precision+recall)))
print("recall is:",recall)
print("precision is:",precision)
print("f1 score is:",f1)

print(matrix)


# In[73]:


review=input()
df = pd.DataFrame()
df['ReviewBody'] = [review]
wordList = textCleaning(df)

   


# In[74]:


vect = count_vect.transform(wordList)
vect=vect.toarray()


# In[75]:


from sklearn.preprocessing import StandardScaler
standardised_data=StandardScaler(with_mean= False ).fit_transform(vect)


# In[76]:


pred = clf_optimal.predict(standardised_data)
print(pred) 


# In[77]:


if pred[0] == 1:
    print("Positive")
else:
    print("Negative")
    

