# import modules

# NLP module
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

# Vectorization module
from sklearn.feature_extraction.text import CountVectorizer

# Topic Models
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation

# packages to store and manipulate data
import pandas as pd
import numpy as np

# plotting packages
import matplotlib.pyplot as plt
import seaborn as sns

# regex
import re

# Saving Model module
import pickle

#Reading tweets dataset and converting it to a DataFrame
df = pd.DataFrame(pd.read_excel('./dataset/Group 6 coke_studio.xlsx'))

# Data Exploration and Visualizations

# make a new column to highlight retweets
df['is_retweet'] = df['tweet'].apply(lambda x: x[:2]=='RT')
df['is_retweet'].sum()  # number of retweets

# Functions to Explore and Visualize Tweet Dataset
def find_retweeted(tweet): #Function to find Retweeted tweets
    '''This function will extract the twitter handles of retweed people'''
    return re.findall('(?<=RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', tweet)

def find_mentioned(tweet): #Function to find Tweet Mentions
    '''This function will extract the twitter handles of people mentioned in the tweet'''
    return re.findall('(?<!RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', tweet)  

def find_hashtags(tweet): #Function to find Tweet Hashtags
    '''This function will extract hashtags'''
    return re.findall('(#[A-Za-z]+[A-Za-z0-9-_]+)', tweet)

# Calling the functions to fing Retweets, Mentions & Hashtags
df['retweeted'] = df.tweet.apply(find_retweeted)
df['mentioned'] = df.tweet.apply(find_mentioned)
df['hashtags'] = df.tweet.apply(find_hashtags)  

# Pipeline to filter hashtags

# take the rows from the hashtag columns where there are actually hashtags
hashtags_df = df.loc[
                       df.hashtags.apply(
                           lambda hashtags_list: hashtags_list !=[]
                       ),['hashtags']]

# create dataframe where each use of hashtag gets its own row
sorted_hashtag_df  = pd.DataFrame(
    [hashtag for hashtags_list in hashtags_df.hashtags
    for hashtag in hashtags_list],
    columns=['hashtag'])

# count of appearances of each hashtag
popular_hashtags = sorted_hashtag_df.groupby('hashtag').size()\
                                        .reset_index(name='counts')\
                                        .sort_values('counts', ascending=False)\
                                        .reset_index(drop=True) 
                                           
# Vectorization
# take hashtags which appear at least this amount of times
min_appearance = 10
# find popular hashtags - make into python set for efficiency
popular_hashtags_set = set(popular_hashtags[
                           popular_hashtags.counts>=min_appearance
                           ]['hashtag'])

# make a new column with only the popular hashtags
hashtags_df['popular_hashtags'] = hashtags_df.hashtags.apply(
            lambda hashtag_list: [hashtag for hashtag in hashtag_list
                                  if hashtag in popular_hashtags_set])
# drop rows without popular hashtag
popular_hashtags_list_df = hashtags_df.loc[
            hashtags_df.popular_hashtags.apply(lambda hashtag_list: hashtag_list !=[])]

# making a Vectorized dataframe
hashtag_vector_df = popular_hashtags_list_df.loc[:, ['popular_hashtags']]

# iterating through the vectorized dataframe
for hashtag in popular_hashtags_set:
    # make columns to encode presence of hashtags
    hashtag_vector_df['{}'.format(hashtag)] = hashtag_vector_df.popular_hashtags.apply(
        lambda hashtag_list: int(hashtag in hashtag_list))

# Dropping Unwanted hashag data column which is unpopular
hashtag_matrix = hashtag_vector_df.drop('popular_hashtags', axis=1)

# CLEANING DATA (STOPWORDS, STEMMING & PUNCTUATION)

#cleaning process variables
my_stopwords = nltk.corpus.stopwords.words('english')
word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'

# Functions to clean tweet
def remove_links(tweet): #function to remove link within tweets
    '''Takes a string and removes web links from it'''
    tweet = re.sub(r'http\S+', '', tweet) # remove http links
    tweet = re.sub(r'bit.ly/\S+', '', tweet) # rempve bitly links
    tweet = tweet.strip('[link]') # remove [links]
    return tweet

def remove_users(tweet): #function to remove a user
    '''Takes a string and removes retweet and @user information'''
    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) # remove retweet
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) # remove tweeted at
    return tweet

def clean_tweet(tweet, bigrams=False): #Function to Clean Tweets
    tweet = remove_users(tweet)
    tweet = remove_links(tweet)
    tweet = tweet.lower() # lower case
    tweet = re.sub('['+my_punctuation + ']+', ' ', tweet) # strip punctuation
    tweet = re.sub('\s+', ' ', tweet) #remove double spacing
    tweet = re.sub('([0-9]+)', '', tweet) # remove numbers
    tweet_token_list = [word for word in tweet.split(' ')
                            if word not in my_stopwords] # remove stopwords

    tweet_token_list = [word_rooter(word) if '#' not in word else word
                        for word in tweet_token_list] # apply word rooter
    if bigrams:
        tweet_token_list = tweet_token_list+[tweet_token_list[i]+'_'+tweet_token_list[i+1]
                                            for i in range(len(tweet_token_list)-1)]
    tweet = ' '.join(tweet_token_list)
    return tweet

#adding a column for clean tweets to the DataFrame
df['clean_tweet'] = df.tweet.apply(clean_tweet)

#VECTORIZATION OF TEXT

#transform text to vector form using vectorizer object
vectorizer = CountVectorizer(max_df=0.9, min_df=25, token_pattern='\w+|\$[\d\.]+|\S+')

# transforming
tf = vectorizer.fit_transform(df['clean_tweet']).toarray()

# Matrix representation
tf_feature_names = vectorizer.get_feature_names_out()

# DEFINING THE MODEL

# LDA model
lda_model = LatentDirichletAllocation(n_components=10, random_state=0)

# NMF model
nmf_model = NMF(n_components=10, random_state=0, alpha_W=.1, l1_ratio=.5)

# Feeding transformed data into the Defined Model using fit method

# Feeding LDA Model
lda_model.fit(tf)

# Feeding NMF Model
# nmf_model.fit(tf)

# DIPLAY TWEET TOPICS

def display_topics(model, feature_names, no_top_words): # Function to Display Tweet Topics
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict) 

# Call Diplay Function

# Display LDA Model generated Topics
# print(display_topics(lda_model, tf_feature_names, 5))

# Display NMF Model generated Topics
# display_topics(nmf_model, tf_feature_names, 5)


# SAVING MODEL
pkl_file = 'pkl_lda_model.pkl' #save lda
# pkl_file = 'pkl_nmf_model.pkl' #save nmf
with open(pkl_file, 'wb') as file:
    pickle.dump(lda_model, file)

# LOADING SAVED MODEL
with open(pkl_file, 'rb') as file:
    saved_lda_model = pickle.load(file) #load lda
    # saved_nmf_model = pickle.load(file) #load nmf

# GENERATING USING THE LOADED MODELcls
top_what = int(input("Enter Number of Topics to Generate :" ))
print(display_topics(saved_lda_model, tf_feature_names, top_what)) # print topics using saved LDA
# display_topics(saved_nmf_model, tf_feature_names,100) # saved NMF

