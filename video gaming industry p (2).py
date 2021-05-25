
# import os
import pandas as pd

# import Dataset 
data = pd.read_csv("C:\\Users\\rohika\\OneDrive\\Desktop\\360digiTMG assignment\\Recomrndation engine\\game.csv", encoding = 'utf8')
data.shape # shape
data.columns
data.game # game columns

from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfid vectorizer 

# replacing the NaN values in overview column with empty string
data["game"].isnull().sum() 
data["game"] = data["game"].fillna(" ")

# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(data.game)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape #12294, 46

# with the above matrix we need to find the similarity score
# There are several metrics for this such as the euclidean, 
# the Pearson and the cosine similarity scores

# For now we will be using cosine similarity matrix
# A numeric quantity to represent the similarity between 2 movies 
# Cosine similarity - metric is independent of magnitude and easy to calculate 

# cosine(x,y)= (x.y⊺)/(||x||.||y||)

from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# creating a mapping of anime name to index number 
data_index = pd.Series(data.index, index = data['game']).drop_duplicates()

data_id = data_index["Metroid Prime"]
data_id

def get_recommendations(Game, topN):    
    # topN = 10
    # Getting the game index using its title 
    data_id = data_index[Game]
    
    
    cosine_scores = list(enumerate(cosine_sim_matrix[data_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar games 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the game index 
    data_idx  =  [i[0] for i in cosine_scores_N]
    data_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar game and scores
    data_similar_show = pd.DataFrame(columns=["game", "Score"])
    data_similar_show["game"] = data.loc[data_idx, "game"]
    data_similar_show["Score"] = data_scores
    data_similar_show.reset_index(inplace = True)  
    # anime_similar_show.drop(["index"], axis=1, inplace=True)
    print (data_similar_show)
    # return (anime_similar_show)

    
# Enter your game and number of games's to be recommended 
get_recommendations("Super Mario Galaxy", topN = 10)
data_index["Perfect Dark"]


