#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv(r"C:\Users\ASUS\Downloads\spam.csv")


# In[3]:


df


# In[4]:


df['target'].value_counts()


# In[5]:


from sklearn.preprocessing import LabelEncoder


# In[6]:


l=LabelEncoder()
df['target']=l.fit_transform(df['target'])


# In[7]:


df


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


X_train,X_test,y_train,y_test=train_test_split(df['text'],df['target'],test_size=0.2,random_state=0)


# In[10]:


X_train.shape


# In[11]:


y_test.shape


# In[12]:


(df['target'].value_counts()/len(df)*100).plot(kind='bar')


# In[13]:


len(df[df['target']==1])


# In[14]:


len(df)


# In[15]:


len(df[df['target']==1])/len(df)*100


# In[16]:


from sklearn.feature_extraction.text import CountVectorizer


# In[17]:


cv=CountVectorizer()


# In[18]:


X_train1=cv.fit_transform(X_train)


# In[19]:


X_train1


# In[20]:


cv


# In[21]:


cv.vocabulary_


# In[22]:


m=max(cv.vocabulary_,key=len)


# In[23]:


m


# In[24]:


from sklearn.naive_bayes import MultinomialNB


# In[25]:


mnb=MultinomialNB(alpha=0.1)


# In[26]:


mnb.fit(X_train1,y_train)


# In[27]:


from sklearn.metrics import roc_auc_score


# In[28]:


X_test1=cv.transform(X_test)


# In[29]:


y_pred=mnb.predict(X_test1)


# In[30]:


roc_auc_score(y_test,y_pred)


# In[31]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[32]:


t=TfidfVectorizer()


# In[33]:


X_train2=t.fit_transform(X_train)
X_test2=t.transform(X_test)


# In[34]:


t.vocabulary_


# In[37]:


t


# In[41]:


len(t.get_feature_names_out())


# In[46]:


names=t.get_feature_names_out()


# In[47]:


names


# In[49]:


tf=X_train2.toarray()


# In[50]:


tf


# In[51]:


max_feature=np.max(tf,axis=0)


# In[52]:


max_feature


# In[54]:


d=pd.Series(max_feature,index=t.get_feature_names_out())


# In[55]:


d


# In[60]:


x=d.sort_values()


# In[61]:


x


# In[62]:


min=x.head(20)


# In[63]:


max=x.tail(20)


# In[64]:


min


# In[65]:


max


# In[66]:


type(min)


# In[70]:


min.groupby(min.index).apply(lambda x: x.sort_index())


# In[71]:


t2=TfidfVectorizer(min_df=3)


# In[72]:


X_train3=t2.fit_transform(X_train)
X_test3=t2.transform(X_test)


# In[74]:


mnb2=MultinomialNB(alpha=0.1)


# In[76]:


mnb2.fit(X_train3,y_train)


# In[77]:


y_pred2=mnb2.predict(X_test3)


# In[78]:


roc_auc_score(y_test,y_pred2)


# In[79]:


df


# In[80]:


spam=df[df['target']==1]


# In[81]:


len(spam)


# In[84]:


from nltk.tokenize import word_tokenize
spam['len']=spam['text'].apply(lambda x : len(word_tokenize(x)))


# In[85]:


spam


# In[86]:


spam['len'].sum()


# In[87]:


df['len']=df['text'].apply(lambda x: len(word_tokenize(x)))


# In[88]:


df


# In[89]:


len_spam=df[df['target']==1]['len'].sum()


# In[90]:


len_spam


# In[91]:


len_ham=df[df['target']==0]['len'].sum()


# In[92]:


len_ham


# In[93]:


avg_spam=len_spam/len(df[df['target']==1])


# In[94]:


avg_ham=len_ham/len(df[df['target']==0])


# In[96]:


avg_spam


# In[97]:


avg_ham


# In[98]:


t3=TfidfVectorizer(min_df=5)


# In[101]:


X_t,X_te,y_T,y_te=train_test_split(df.drop(columns=['len','target']),df['target'],test_size=0.2,random_state=0)


# In[103]:


t3


# In[104]:


X_t


# In[105]:


X_t=t3.fit_transform(X_train)


# In[106]:


X_te=t3.transform(X_test)


# In[108]:


from sklearn.svm import SVC


# In[109]:


s=SVC(C=10000)


# In[110]:


s


# In[112]:


def add_feature(X,new):
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# In[115]:


s.fit(X_t,y_train)


# In[116]:


y_pred3=s.predict(X_te)


# In[117]:


roc_auc_score(y_test,y_pred3
            )


# In[118]:


df


# In[119]:


import re


# In[120]:


pattern=re.compile(r'\d')


# In[121]:


df['digit']=df['text'].apply(lambda x: len(re.findall(pattern,x)))


# In[122]:


df


# In[123]:


df.head(3).values


# In[124]:


ls=len(df[df['target']==1])
lh=len(df[df['target']==0])


# In[125]:


ls


# In[126]:


lh


# In[127]:


spam_sum_digit=df[df['target']==1]['digit'].sum()


# In[128]:


ham_sum_digit=df[df['target']==0]['digit'].sum()


# In[130]:


spam_sum_digit/ls


# In[131]:


ham_sum_digit/lh


# In[132]:


df


# In[136]:


tf=TfidfVectorizer(min_df=5,ngram_range=(1,3))


# In[137]:


tf


# In[143]:


X=tf.fit_transform(df['text'])


# In[146]:


X.shape


# In[147]:


extra=np.array(df[['len','digit']])


# In[148]:


extra


# In[149]:


extra.shape


# In[152]:


extra=np.array(extra)


# In[153]:


extra


# In[154]:


X=np.array(X)


# In[155]:


X


# In[158]:


from scipy.sparse import csr_matrix


# In[160]:


from scipy.sparse import hstack


# In[163]:


df['vect']=X


# In[164]:


df


# In[167]:


from sklearn.linear_model import LogisticRegression


# In[168]:


lr=LogisticRegression(C=100,max_iter=1000)


# In[171]:


def add_feature(X, feature_to_add):
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# In[172]:


X


# In[178]:


df.drop(columns='vect',inplace=True)


# In[179]:


df


# In[180]:


vec=TfidfVectorizer(min_df=5,ngram_range=(1,3))


# In[181]:


vec


# In[184]:


X=df.drop(columns='target')


# In[185]:


y=df['target']


# In[186]:


X


# In[197]:


xtrain,xtest,train,ytest=train_test_split(df.drop(columns=['len','digit','target']),df['target'],test_size=0.2,random_state=0)


# In[198]:


xtrain


# In[202]:


pattern2=re.compile(r'[^_\w]')
df['non-word']=df['text'].apply(lambda x: len(re.findall(pattern2,x)))


# In[203]:


df


# In[204]:


spam_sum=df[df['target']==1]['non-word'].sum()


# In[205]:


spam_sum


# In[206]:


ham_sum=df[df['target']==0]['non-word'].sum()


# In[207]:


ham_sum


# In[208]:


spam_sum/ls


# In[209]:


ham_sum/lh


# In[210]:


X_train


# In[211]:


y_train


# In[214]:


CV=CountVectorizer(min_df=5,ngram_range=(2,5),analyzer='char_wb')


# In[215]:


CV


# In[216]:


l=LogisticRegression(C=100,max_iter=1000)


# In[217]:


X_train=CV.fit_transform(X_train)
X_test=CV.transform(X_test)


# In[218]:


X_train


# In[220]:


from sklearn.metrics import roc_auc_score


# In[221]:


l.fit(X_train,y_train)


# In[222]:


roc_auc_score(y_test,l.predict(X_test))


# In[224]:


feature_names=CV.get_feature_names_out()


# In[226]:


feature_names


# In[229]:


m=l.coef_[0].argsort()


# In[230]:


smallest=feature_names[m[:10]]


# In[233]:


largest=feature_names[m[:-11:-1]]


# In[234]:


smallest


# In[235]:


largest

