#Prompt user to enter a title and then use embeddings to recommend movies/series that have
#a similar synopsis and are available in Netflix South Arica
#Gillian Metcalf 29/06/2023

import streamlit as st 
import openai
import pandas as pd
import numpy as np

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(
    page_title='Netflix South African Recomender', 
    page_icon=":film_projector:", 
    layout="centered", 
    initial_sidebar_state="auto", 
    menu_items=None)

st.header(':red[Netflix South African Recomender] :film_projector:')

#Import file and convert embedding to array
@st.cache_data
def load_data():
    
    file_url = 'https://drive.google.com/uc?id=1rU1iKFZVfQ4GLbgc7Dozs8N4Tw7YTkU4'
    df_embed = pd.read_csv(file_url)
    #df_embed=pd.read_csv('netflixRsaEmbeddings.csv',delimiter=',')
    df_embed['embedding']=df_embed['embedding'].apply(eval).apply(np.array) #converts embedding to numpy array
    return df_embed

#Get recomendation from title
def get_recomdendation_from_title(df_embeddings, title, k):
    from openai.embeddings_utils import distances_from_embeddings, indices_of_nearest_neighbors_from_distances
    index = 0
    if title not in list(df_embeddings['title']):
        return False
            
    movie_embedding = df_embeddings[df_embeddings['title'] == title]['embedding']
    movie_embedding = movie_embedding.squeeze()    #Converts into a Python list  -  embeddings for selected title
    
    embeddings = list(df_embeddings['embedding'])    #List of all embeddings
    
    distances = distances_from_embeddings(movie_embedding,embeddings)    
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)
        
    recomendations = list()

    indices = indices_of_nearest_neighbors[0:300]
    st.write(df_embeddings.iloc[indices[0]]['synopsis'] )      #description of entered movie
    st.divider()
           
    count = 1

    #while index < len(indices):
    while ((index < k) and (count < len(indices))):
        current_index = indices[count]
        if ((cb_movies and df_embeddings.iloc[current_index]['title_type'] == 'movie' ) or (cb_series and df_embeddings.iloc[current_index]['title_type'] == 'series' )): 
            movie = dict()    #Python dictionary
            movie['Title'] = df_embeddings.iloc[current_index]['title']      #title
            movie['Description'] = df_embeddings.iloc[current_index]['synopsis']      #description
            movie['Type'] = df_embeddings.iloc[current_index]['title_type']      #Movie / Series
            movie['Distance'] = distances[current_index]
            recomendations.append(movie)
            index += 1
        count += 1
    return recomendations     
    
st.text_input("Enter the movie/series title: ",key="title")

cb_movies = st.checkbox('Movies',value=True)
cb_series = st.checkbox('Series',value=True)

if st.button(':mag:'):

    df_embed = load_data()

    #Run recomendation function
    movie_recomendations= get_recomdendation_from_title(df_embed, st.session_state.title, 10)

    if movie_recomendations:
        
        for i, item in enumerate(movie_recomendations):
                        
            st.write(f'Title: {item["Title"]}')
            st.write(f'Description: {item["Description"]}') 
            st.write(f'Type: {item["Type"]}')             
            st.divider()
    else:
     st.write(f'{st.session_state.title} not in dataset')       

