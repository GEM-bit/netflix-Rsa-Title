#Prompt user to enter a title and then use embeddings to recommend movies/series that have
#a similar synopsis and are available in Netflix South Arica
#Gillian Metcalf 29/06/2023

#Changes
#Store vectors in ChromaDB instead of Pinecone
#Add link to Rotten Tomatoes for each recomendation
#Put suggestions in st.text_area

import streamlit as st 
import openai

import pandas as pd
import numpy as np

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(
    page_title='​Personalized NetflixSA', 
    page_icon=":film_projector:", 
    layout="centered", 
    initial_sidebar_state="auto", 
    menu_items=None)

st.header(':red[​Personalized NetflixSA:]:film_projector:')
st.subheader(':red[Uncover Content Just for You] ')

#Import file and convert embedding to array
@st.cache_data(persist=True)
def load_data():
    
    try:
        df_embed1 = pd.read_csv('netflixrsaembeddings.csv',delimiter=',')
        df_embed2 = pd.read_csv('output1.csv',delimiter=',')
        df_embed3 = pd.read_csv('output3.csv',delimiter=',')
        df_embed4 = pd.read_csv('output4.csv',delimiter=',')
        df_embed5 = pd.read_csv('output5.csv',delimiter=',')

        df_embed = pd.concat([df_embed1, df_embed2,df_embed3,df_embed4,df_embed5], ignore_index=True)
        #df_embed = pd.concat([df_embed1, df_embed2], ignore_index=True)

        #df_embed = pd.read_csv(file_url)
        print("CSV file read successfully.")
    except Exception as e:
        print("An error occurred while reading the CSV file:", str(e)) 

    df_embed['embedding']=df_embed['embedding'].apply(eval).apply(np.array) #converts embedding to numpy array
    return df_embed

#Get recomendation from title
def get_recomdendation_from_title(df_embeddings, title, k):
    from langchain.embeddings import OpenAIEmbeddings
    from OpenAIEmbeddings import distances_from_embeddings, indices_of_nearest_neighbors_from_distances
    index = 0
    
    if title.lower().rstrip() not in map(str.strip, df_embeddings['title'].str.lower()):
        return False
            
    movie_embedding = df_embeddings[df_embeddings['title'].str.lower().str.strip() == title.lower().rstrip()]['embedding']
        
    movie_embedding = movie_embedding.squeeze()    #Converts into a Python list  -  embeddings for selected title
    
    embeddings = list(df_embeddings['embedding'])    #List of all embeddings
    
    distances = distances_from_embeddings(movie_embedding,embeddings)    
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)
        
    recomendations = list()

    indices = indices_of_nearest_neighbors[0:300]
    st.text_area(label='Synopsis',value=df_embeddings.iloc[indices[0]]['synopsis'],height=100 )      #description of entered movie
    st.divider()
           
    count = 1

    while ((index < k) and (count < len(indices))):
        current_index = indices[count]
        if ((cb_movies and df_embeddings.iloc[current_index]['title_type'] == 'movie' ) or (cb_series and df_embeddings.iloc[current_index]['title_type'] == 'series' )): 
            movie = dict()    #Python dictionary
            movie['Title'] = df_embeddings.iloc[current_index]['title']      #title
            movie['Description'] = df_embeddings.iloc[current_index]['synopsis']      #description
            movie['Type'] = df_embeddings.iloc[current_index]['title_type']      #Movie / Series
            movie['Year'] = df_embeddings.iloc[current_index]['year']      #Movie / Series
            movie['Distance'] = distances[current_index]
            recomendations.append(movie)
            index += 1
        count += 1
    return recomendations     

# ########################################################## Main Code ###############################################################    
enteredTitle = st.text_input("Enter the movie/series title: ",key="title")

cb_movies = st.checkbox('Movies',value=True)
cb_series = st.checkbox('Series',value=True)

if enteredTitle != '':               #Title and enter button clicked
    if st.session_state.get('last_text') != enteredTitle:    
        df_embed = load_data()

        #Run recomendation function
        movie_recomendations= get_recomdendation_from_title(df_embed, st.session_state.title, 11)

        if movie_recomendations:

            for i, item in enumerate(movie_recomendations):

                searchString = f'{item["Title"]}%20{item["Year"]}%20{item["Type"]}'
                searchString=searchString.replace(" ", "%20")
                link=f'https://www.rottentomatoes.com/search?search={searchString}'

                text1=f'## Title: {item["Title"]}'
                text2=f'''Description: {item["Description"]}'''
                text3=f'Year: {item["Year"]}'
                text4=f'Type: {item["Type"]}'
                text5=f'[Review Link]({link})'
                            
                st.markdown(text1)
                st.markdown(text2)
                st.markdown(text3)
                st.markdown(text4)
                st.markdown(text5)
            st.session_state['last_text'] = enteredTitle
    else:
        st.write(f'{st.session_state.title} not in dataset')      

# run the app: streamlit run ./netflixRecommend.py

