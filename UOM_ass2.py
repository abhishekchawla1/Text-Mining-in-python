#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import pandas as pd
import numpy as np


# In[2]:


path = 'C:\\Users\\ASUS\\Downloads\\plots.txt'

with open(path,'r',encoding='utf-8') as f:
    data=f.read()
    


# In[3]:


data


# In[4]:


from nltk.tokenize import word_tokenize


# In[5]:


len(word_tokenize(data))


# In[6]:


len(set(word_tokenize(data.lower())))


# In[7]:


from nltk.stem import WordNetLemmatizer
l=WordNetLemmatizer()
lem=[l.lemmatize(w,'v') for w in word_tokenize(data)]
len(set(lem))


# In[8]:


def words(text):
    tokens=word_tokenize(data)
    print(f'Total tokens in the text: {len(tokens)}')
    print(f'Total Unique Words in the text: {len(set(tokens))}')
    lem=[l.lemmatize(w) for w in tokens]
    print(f'Total unique words in lemmatized text: {len(set(lem))}')


# In[9]:


words(data)


# In[10]:


#lexical_Diversity=unique tokens/total tokens


# In[11]:


def lex_div(text):
    words=len(word_tokenize(text))
    unique=len(set(word_tokenize(text)))
    lex=unique/words
    return lex    


# In[12]:


lex_div(data)


# In[18]:


def perc(text):
    li=['love','Love']
    c1=0
    c2=0
    import string
    str=''
    p=string.punctuation
    for x in text:
        if x not in p:
            str=str+x
    text=str
    for word in word_tokenize(text):
        if word in li:
            c1+=1
    for word in word_tokenize(text.lower()):
        if word in li:
            c2+=1
    return (c1,c2)


# In[19]:


perc(data)


# In[20]:


#What percentage of words is love or Love?


# In[21]:


per=(463*100)/len(word_tokenize(data))


# In[22]:


per


# In[23]:


#20 Most occouring tokens in the text


# In[34]:


from nltk.probability import FreqDist
def freq(text):
    dist=FreqDist(word_tokenize(text))
    new_dict=dict(sorted(dist.items(),key=lambda x: x[1],reverse=True))
    return new_dict


# In[29]:


FreqDist(word_tokenize(data))


# In[31]:


new=dict(sorted(FreqDist(word_tokenize(data)).items(),key=lambda x:x[1],reverse=True))


# In[35]:


new


# In[37]:


FreqDist(word_tokenize(data)).most_common(20)


# In[39]:


from wordcloud import WordCloud


# In[41]:


wc=WordCloud(height=500,width=500,background_color='white')


# In[43]:


import matplotlib.pyplot as plt
plt.imshow(wc.generate_from_frequencies(dict(FreqDist(word_tokenize(data)).most_common(20))))


# In[44]:


#tokens with len>5 and freq>200


# In[52]:


def answer(text):
    tokens=word_tokenize(text)
    dictionary=FreqDist(tokens)
    li=[]
    for word,freq in dictionary.items():
        if len(word)>5 and freq>200:
            li.append(word)
    return li


# In[55]:


x=answer(data)


# In[56]:


sorted(x)


# In[80]:


#Longest word in the text


# In[69]:


def longest(text):
    text=nltk.Text(word_tokenize(text))
    long=max(FreqDist(text),key=len)
    return (long,len(long))


# In[70]:


text='Hi my name is Abhishek'
tokens=word_tokenize(text)
text=nltk.Text(tokens)


# In[71]:


text


# In[72]:


long=max(FreqDist(text),key=len)


# In[73]:


long


# In[74]:


len(long)


# In[75]:


longest(data)


# In[81]:


#unique words with freq>2000
def occou_words(text):
    tokens=word_tokenize(text.lower())
    li=[]
    dist=FreqDist(tokens)
    for word,freq in dist.items():
        if word.isalpha():
            if freq>2000:
                li.append((freq,word))
    return li
    
    


# In[82]:


occou_words(data)


# In[84]:


wc=WordCloud(height=500,width=500,background_color='white',min_font_size=5)


# In[87]:


flipped_dict = {value: key for key, value in dict(occou_words(data)).items()}
plt.imshow(wc.generate_from_frequencies(flipped_dict))


# In[88]:


d=nltk.Text(word_tokenize(data))


# In[89]:


d


# In[90]:


from nltk.tokenize import sent_tokenize


# In[97]:


string_text=' '.join(tokens)    


# In[98]:


string_text


# In[99]:


#average number of whitespace separated tokens per sentence in the sentence-tokenized copy of data


# In[100]:


def res(text):
    sentences=sent_tokenize(text)
    words=word_tokenize(text)
    return len(words)/len(sentences)


# In[101]:


res(data)


# In[102]:


sent_tokenize(data)[1:5]


# In[103]:


from nltk import pos_tag


# In[119]:


#5 most common pos

def part_of_speech(text):
    p=pos_tag(word_tokenize(text))
    li=[]
    for name,entity in p:
        li.append(entity)
    from collections import Counter
    return Counter(li).most_common(5)


# In[118]:


part_of_speech(data)


# In[110]:


x=pos_tag(word_tokenize(data))


# In[114]:


from collections import Counter
l=[]
for n,e in x:
    l.append(e)
Counter(l).most_common(10)

