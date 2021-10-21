import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import contractions
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim
import nltk
from dateutil.parser import parse
import streamlit.components.v1 as components
from datetime import datetime
from datetime import date

@st.cache
def load_data():

    scrob = pd.read_csv('data/scrobb.csv')
    albumslistened_all = pd.read_csv('data/scrobb_all.csv')
    albumslistened_week = pd.read_csv('data/scrobb_week.csv')

    wants = pd.read_csv('data/wants.csv')
    collection = pd.read_csv('data/collection.csv')
    ratingsfinal = pd.read_csv('data/ratingsfinal.csv')

    metratingsn = pd.read_csv('data/metratings.csv')
    metmatchalbss = pd.read_csv('data/metmatch.csv')

    pitch = pd.read_csv('data/pitch_weights.csv')
    pitchart = pd.read_csv('data/pitchmatchart.csv')

    return scrob, albumslistened_all, albumslistened_week, wants, collection, ratingsfinal, metratingsn, metmatchalbss, pitch, pitchart

scrob, albumslistened_all, albumslistened_week, wants, collection, ratingsfinal, metratingsn, metmatchalbss, pitch, pitchart = load_data()




st.sidebar.title("Music APP")
page = st.sidebar.radio(
     "Pick an option",
     ('Home' , 'Habits', 'Discography', 'Suggestor - AOTY', 'Suggestor - Pitch'),
     )

if page == "Home":

    st.header("Overview")
    st.markdown("Purpose of this app is to:")
    st.markdown("* Outline my listening habits")
    st.markdown("* Give an overview of Vinyl Discography - Collected/Wanted/Rated")
    st.markdown("* Provide an album suggestor")

elif page == "Habits":

    time = st.radio(
         "Pick a Time",
         ('All-Time', 'Weekly'),
         )
    if time== "All-Time":
        col1, col2 = st.columns(2)
        with col1:
            st.header("Artists")
            st.image('./cloud/artistall.png', use_column_width='always')
            st.markdown("Top Artist album listens")
            albumslistened_all.Artist.value_counts()[:9]
        with col2:
            st.header("Albums")
            st.image('./cloud/albumsall.png', use_column_width='always')
            st.markdown("Top Album listens")
            albumslistened_all.Title.value_counts()[:9]
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.header("Artists")
            st.image('./cloud/artistweek.png', use_column_width='always')
            st.markdown("Top Artist album listens this week")
            albumslistened_week.Artist.value_counts()[:5]
        with col2:
            st.header("Albums")
            st.image('./cloud/albumsweek.png', use_column_width='always')
            st.markdown("Top Album listens this week")
            albumslistened_week.Title.value_counts()[:5]

elif page== "Discography":

    st.subheader("Discography Page")

    cat = st.radio(
         "Pick a Category",
         ('Wanted', 'Collected', 'Rated'),
         )

    if cat== "Wanted":
        st.image('./cloud/artistwanted.png', use_column_width='always')
        col1, col2 = st.columns(2)
        with col1:
            wants.Artist.value_counts()[:9]
        with col2:
            wants.Artist.value_counts()[10:19]


    elif cat == "Collected":
        st.image('./cloud/artistcollected.png', use_column_width='always')
    else:
        st.image('./cloud/albumrating5.png', use_column_width='always')
        st.image('./cloud/albumrating4.png', use_column_width='always')
        st.image('./cloud/albumrating3.png', use_column_width='always')


elif page== "Suggestor - AOTY":

    st.subheader("AOTY Suggestor Page")
    st.write("This page uses a K-Means Clustering technique on a 30K + data set of AOTY Critic and User Scores.")
    st.write("Metacritic scores are also present but missing, so the primary inputs are genre, AOTY review numbers and scores, and years released.")
    k = list(metmatchalbss.Artist.unique())
    y = st.selectbox('Pick an artist', sorted(k))


    a = list(metratingsn.loc[metratingsn.Artist == y].Title.unique())
    q = st.selectbox('Pick an Album', sorted(a))

    minim = metratingsn.loc[metratingsn.Title == q]['kclusters'].values[0]

    clusterrats = metratingsn.loc[metratingsn.kclusters == minim]
    reviews = metratingsn.loc[(metratingsn.Title == q) & (metratingsn.Artist == y)][['AOTYCriticScore', 'AOTYUserScore', 'MetacriticCriticScore','MetacriticUserScore',]]
    st.write("This Albums Reviews:")
    reviews

    st.write("_______________________________")

    back = st.radio(
         "Pick a Metric",
         ('Critic AOTY', 'User AOTY'),
         )
    if back == 'Critic AOTY':
        st.write("Critic Scores Sorted")
        options = st.multiselect( 'Tune the clusters up',  ['Same Year', 'Same Genre', 'Relative Score'], ['Same Year'])

        year = metratingsn.loc[(metratingsn.Title == q) & (metratingsn.Artist == y)]['year'].values[0]
        genre = metratingsn.loc[(metratingsn.Title == q) & (metratingsn.Artist == y)]['Genre'].values[0]

        relativelow = metratingsn.loc[(metratingsn.Title == q) & (metratingsn.Artist == y)]['AOTYCriticScore'].values[0] - 10
        relativehi = metratingsn.loc[(metratingsn.Title == q) & (metratingsn.Artist == y)]['AOTYCriticScore'].values[0] + 10

        if options == ['Same Year']:
            clusterrats = clusterrats.loc[clusterrats.year == year]
            fin = clusterrats.sort_values(by=['AOTYCriticScore'], ascending=False)[:100][['Artist', 'Title', 'year', 'AOTYCriticScore']]
            l = fin.sample(5, replace=True)
            st.dataframe(l)
        elif options == ['Same Genre']:
            clusterrats = clusterrats.loc[clusterrats['Genre'] == genre]
            fin = clusterrats.sort_values(by=['AOTYCriticScore'], ascending=False)[:100][['Artist', 'Title', 'year', 'AOTYCriticScore']]
            l = fin.sample(5, replace=True)
            st.dataframe(l)
        elif options == ['Relative Score']:
            clusterrats = clusterrats.loc[(clusterrats.AOTYCriticScore <= relativehi) & (clusterrats.AOTYCriticScore >= relativelow)]
            fin = clusterrats.sort_values(by=['AOTYCriticScore'], ascending=False)[:100][['Artist', 'Title', 'year', 'AOTYCriticScore']]
            l = fin.sample(5, replace=True)
            st.dataframe(l)
        elif options == ['Same Year', 'Same Genre',]:
            clusterrats = clusterrats.loc[(clusterrats.year == year) & (clusterrats['Genre'] == genre)]
            fin = clusterrats.sort_values(by=['AOTYCriticScore'], ascending=False)[:100][['Artist', 'Title', 'year', 'AOTYCriticScore']]
            l = fin.sample(5, replace=True)
            st.dataframe(l)
        elif options == ['Same Year', 'Same Genre','Relative Score']:
            clusterrats = clusterrats.loc[(clusterrats.year == year) & (clusterrats['Genre'] == genre) & (clusterrats.AOTYCriticScore <= relativehi) & (clusterrats.AOTYCriticScore >= relativelow)]
            fin = clusterrats.sort_values(by=['AOTYCriticScore'], ascending=False)[:100][['Artist', 'Title', 'year', 'AOTYCriticScore']]
            l = fin.sample(5, replace=True)
            st.dataframe(l)
        elif options == [ 'Same Genre','Relative Score']:
            clusterrats = clusterrats.loc[(clusterrats['Genre'] == genre) & (clusterrats.AOTYCriticScore <= relativehi) & (clusterrats.AOTYCriticScore >= relativelow)]
            fin = clusterrats.sort_values(by=['AOTYCriticScore'], ascending=False)[:100][['Artist', 'Title', 'year', 'AOTYCriticScore']]
            l = fin.sample(5, replace=True)
            st.dataframe(l)


    elif back == 'User AOTY':

        st.write("User Scores Sorted")
        options = st.multiselect( 'Tune the clusters up',  ['Same Year', 'Same Genre', 'Relative Score'], ['Same Year'])

        year = metratingsn.loc[metratingsn.Title == q]['year'].values[0]
        genre = metratingsn.loc[metratingsn.Title == q]['Genre'].values[0]

        relativelow = metratingsn.loc[metratingsn.Title == q]['AOTYUserScore'].values[0] - 10
        relativehi = metratingsn.loc[metratingsn.Title == q]['AOTYUserScore'].values[0] + 10

        if options == ['Same Year']:
            clusterrats = clusterrats.loc[clusterrats.year == year]
            fin = clusterrats.sort_values(by=['AOTYUserScore'], ascending=False)[:100][['Artist', 'Title', 'year', 'AOTYUserScore']]
            l = fin.sample(5, replace=True)
            st.dataframe(l)
        elif options == ['Same Genre']:
            clusterrats = clusterrats.loc[clusterrats['Genre'] == genre]
            fin = clusterrats.sort_values(by=['AOTYUserScore'], ascending=False)[:100][['Artist', 'Title', 'year', 'AOTYUserScore']]
            l = fin.sample(5, replace=True)
            st.dataframe(l)
        elif options == ['Relative Score']:
            clusterrats = clusterrats.loc[(clusterrats.AOTYCriticScore <= relativehi) & (clusterrats.AOTYCriticScore >= relativelow)]
            fin = clusterrats.sort_values(by=['AOTYUserScore'], ascending=False)[:100][['Artist', 'Title', 'year', 'AOTYUserScore']]
            l = fin.sample(5, replace=True)
            st.dataframe(l)
        elif options == ['Same Year', 'Same Genre',]:
            clusterrats = clusterrats.loc[(clusterrats.year == year) & (clusterrats['Genre'] == genre)]
            fin = clusterrats.sort_values(by=['AOTYUserScore'], ascending=False)[:100][['Artist', 'Title', 'year', 'AOTYUserScore']]
            l = fin.sample(5, replace=True)
            st.dataframe(l)
        elif options == ['Same Year', 'Same Genre','Relative Score']:
            clusterrats = clusterrats.loc[(clusterrats.year == year) & (clusterrats['Genre'] == genre) & (clusterrats.AOTYCriticScore <= relativehi) & (clusterrats.AOTYCriticScore >= relativelow)]
            fin = clusterrats.sort_values(by=['AOTYUserScore'], ascending=False)[:100][['Artist', 'Title', 'year', 'AOTYUserScore']]
            l = fin.sample(5, replace=True)
            st.dataframe(l)
        elif options == [ 'Same Genre','Relative Score']:
            clusterrats = clusterrats.loc[(clusterrats['Genre'] == genre) & (clusterrats.AOTYCriticScore <= relativehi) & (clusterrats.AOTYCriticScore >= relativelow)]
            fin = clusterrats.sort_values(by=['AOTYUserScore'], ascending=False)[:100][['Artist', 'Title', 'year', 'AOTYUserScore']]
            l = fin.sample(5, replace=True)
            st.dataframe(l)

elif page== "Suggestor - Pitch":
    st.subheader("Pitchfork Suggestor Page")
    pitch = pitch.dropna()

    st.write("This page uses a Latent Dirichlet Allocation technique on a 27K + data set of Pitchfork Written Reviews.")

    k = list(pitchart.artist.unique())
    y = st.selectbox('Pick an artist', sorted(k))


    a = list(pitch.loc[pitch.artist == y].album.unique())
    q = st.selectbox('Pick an Album', sorted(a))
    toptop = pitch.loc[(pitch.album == q) & (pitch.artist == y)]['toptop'].values[0]


    minim = pitch.loc[pitch.album == q]['toptop'].values[0]
    pitch['year'] = pd.to_datetime(pitch['date']).dt.year

    clusterrats = pitch.loc[pitch.toptop == minim]
    reviews = pitch.loc[pitch.album == q][['score', 'genre', toptop,'year']]
    st.write("This Albums Reviews:")
    reviews

    st.write("_______________________________")

    st.write("")
    options = st.multiselect( 'Tune the LDA up',  ['Same Year', 'Same Genre', 'Relative Score'], ['Same Year'])

    year = pitch.loc[pitch.album == q]['year'].values[0]
    genre = pitch.loc[pitch.album == q]['genre'].values[0]

    relativelow = pitch.loc[pitch.album == q]['score'].values[0] - 1.5
    relativehi = pitch.loc[pitch.album == q]['score'].values[0] + 1.5
    merged = ['score', toptop]
    if options == ['Same Year']:
        clusterrats = clusterrats.loc[clusterrats.year == year]
        fin = clusterrats.sort_values(by=merged, ascending=False)[:100][['artist', 'album', 'year', 'score', toptop]]
        l = fin.sample(5, replace=True)
        st.dataframe(l)
    elif options == ['Same Genre']:
        clusterrats = clusterrats.loc[clusterrats['genre'] == genre]
        fin = clusterrats.sort_values(by=merged, ascending=False)[:100][['artist', 'album', 'year', 'score', toptop]]
        l = fin.sample(5, replace=True)
        st.dataframe(l)
    elif options == ['Relative Score']:
        clusterrats = clusterrats.loc[(clusterrats.score <= relativehi) & (clusterrats.score >= relativelow)]
        fin = clusterrats.sort_values(by=merged, ascending=False)[:1000][['artist', 'album', 'year', 'score', toptop]]
        l = fin.sample(5, replace=True)
        st.dataframe(l)
    elif options == ['Same Year', 'Same Genre',]:
        clusterrats = clusterrats.loc[(clusterrats.year == year) & (clusterrats['genre'] == genre)]
        fin = clusterrats.sort_values(by=merged, ascending=False)[:100][['artist', 'album', 'year', 'score', toptop]]
        l = fin.sample(5, replace=True)
        st.dataframe(l)
    elif options == ['Same Year', 'Same Genre','Relative Score']:
        clusterrats = clusterrats.loc[(clusterrats.year == year) & (clusterrats['genre'] == genre) & (clusterrats.score <= relativehi) & (clusterrats.score >= relativelow)]
        fin = clusterrats.sort_values(by=merged, ascending=False)[:1000][['artist', 'album', 'year', 'score', toptop]]
        l = fin.sample(5, replace=True)
        st.dataframe(l)
    elif options == [ 'Same Genre','Relative Score']:
        clusterrats = clusterrats.loc[(clusterrats['genre'] == genre) & (clusterrats.score <= relativehi) & (clusterrats.score >= relativelow)]
        fin = clusterrats.sort_values(by=merged, ascending=False)[:1000][['artist', 'album', 'year', 'score', toptop]]
        l = fin.sample(5, replace=True)
        st.dataframe(l)
