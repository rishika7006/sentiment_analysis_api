import warnings
from datetime import datetime, timedelta

import numpy as np
import seaborn as sns
import streamlit as st
from google_play_scraper import Sort, reviews
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")

import nltk
import pandas as pd
# from nltk.corpus import stopwords
# from textblob import TextBlob

from textblob import Word

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
# Importing Seaborn and Matplotlib for graphical effects.

# To Hide Warnings
# st.set_option('deprecation.showfileUploaderEncoding', False)
# st.set_option('deprecation.showPyplotGlobalUse', False)
# Viz Pkgs
import matplotlib

matplotlib.use('Agg')
# sns.set_style('darkgrid')
STYLE = """
<style>
img {
    max-width: 100%;
}
</style> """

# def main():
""" Common ML Dataset Explorer """
# st.title("Live twitter Sentiment analysis")
# st.subheader("Select a topic which you'd like to get the sentiment analysis on :")

html_temp = """
<div style="background-color:tomato;"><p style="color:white;font-size:40px;padding:9px">Live twitter Sentiment analysis</p></div>
"""
st.markdown(html_temp, unsafe_allow_html=True)
st.subheader("Results for past 7 days")

# Write a code to extract reviews for past 1 week
result, continuation_token = reviews(
    'com.freedomrewardz',  # found in app's url
    lang='en',  # defaults to 'en'
    country='us',  # defaults to 'us'
    sort=Sort.NEWEST,  # start with most recent
    count=500  # batch size
)
df = pd.DataFrame(result)
df['at'] = pd.to_datetime(df['at'])
start_date = datetime.today() - timedelta(days=7)
end_date = datetime.today()
mask = (df['at'] > start_date) & (df['at'] <= end_date)
df = df.loc[mask]
df.drop(['repliedAt', 'userImage'], axis='columns', inplace=True)
x = df['reviewId'].count()
# Show the dimension of the dataframe
if st.checkbox("Show number of rows and columns"):
    st.write(f'Rows: {df.shape[0]}')
    st.write(f'Columns: {df.shape[1]}')

# display the dataset
if st.checkbox("Show Dataset"):
    st.write("### Enter the number of rows to view")
    rows = st.number_input("", min_value=0, value=5)
    if rows > 0:
        st.dataframe(df.head(rows))

# Text Preprocessing
# Lower casing:
df['content'] = df['content'].str.lower()
# Removing punctuations:
df['content'] = df['content'].str.lower()
# Lemmatization
df['content'] = df['content'].apply(lambda y: " ".join([Word(word).
                                                       lemmatize() for word in y.split()]))
# replace
df['content'] = df['content'].str.replace("rewardz", "reward")

# get the countPlot
if st.button("Get Number of reviews for each Score"):
    st.success("Generating A Count Plot")
    st.subheader(" Count Plot for Different Scores")
    st.write(sns.countplot(df["score"]))
    st.pyplot()

# full EDA
if st.checkbox("Visualize Columns wrt Classes"):
    st.write("#### Select column to visualize: ")
    columns = df.columns.tolist()
    class_name = columns[-1]
    column_name = st.selectbox("", columns)
    st.write("#### Select type of plot: ")
    plot_type = st.selectbox("", ["kde", "box", "violin", "swarm"])
    if st.button("Generate"):
        if plot_type == "kde":
            st.write(
                sns.FacetGrid(df, hue=class_name, palette="husl", height=6).map(sns.kdeplot, column_name).add_legend())
            st.pyplot()

        if plot_type == "box":
            st.write(sns.boxplot(x=class_name, y=column_name, palette="husl", data=df))
            st.pyplot()

        if plot_type == "violin":
            st.write(sns.violinplot(x=class_name, y=column_name, palette="husl", data=df))

            st.pyplot()
        if plot_type == "swarm":
            st.write(sns.swarmplot(x=class_name, y=column_name, data=df, color="y", alpha=0.9))
            st.pyplot()

# Function for getting the sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
sentiment_comp = []
for row in df['content']:
    vs = analyzer.polarity_scores(row)
    sentiment_comp.append(vs)

# Creating new dataframe with sentiments
df_sentiments = pd.DataFrame(sentiment_comp)
# Merging the sentiments back to reviews dataframe
df = pd.concat([df.reset_index(drop=True), df_sentiments], axis=1)
percentage_comp = []

for i in df['compound']:
    if i > 0:
        percentage = i * 100
    elif i < 0:
        percentage = -i * 100
    else:
        percentage = " "
    percentage_comp.append(percentage)

df['percentage'] = percentage_comp
# Convert scores into positive, negative and not defined sentiments using some threshold
df["Sentiment"] = df["compound"].apply(lambda compound: "positive" if compound > 0 else \
    ("negative" if compound < 0 else "not defined"))
df.drop(['neg', 'neu', 'pos', 'compound'], axis='columns', inplace=True)
st.write('Check out the sentiments with percentage of positivity/negativity')
st.success('doing Sentiment Analysis')
# Select columns to display
if st.checkbox("Show dataset with selected columns"):
    # get the list of columns
    columns = df.columns.tolist()
    st.write("#### Select the columns to display:")
    selected_cols = st.multiselect("", columns)
    if len(selected_cols) > -1:
        selected_df = df[selected_cols]
        st.dataframe(selected_df)

# get the countPlot
if st.button("How many Positive, Negative and Neutral reviews?"):
    st.success("Generating A Count Plot")
    st.subheader(" Count Plot for Different Sentiments")
    st.write(sns.countplot(df["Sentiment"]))
    st.pyplot()
# Piechart
if st.button("Get Pie Chart for Different Sentiments"):
    st.success("Generating A Pie Chart")
    a = len(df[df["Sentiment"] == "Positive"])
    b = len(df[df["Sentiment"] == "Negative"])
    c = len(df[df["Sentiment"] == "Neutral"])
    d = np.array([a, b, c])
    explode = (0.1, 0.0, 0.1)
    st.write(plt.pie(d, shadow=True, explode=explode, labels=["Positive", "Negative", "Neutral"], autopct='%1.2f%%'))
    st.pyplot()
# Build wordcloud:
# calculating sentiments
reviews = np.array(df['content'])
size = (len(df))

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
pos_reviews = ''
neg_reviews = ''
concat_reviews = ''
score_arr = []
for i in range(size):
    sentence = reviews[i]
    concat_reviews += ' %s' % sentence
    vs = analyzer.polarity_scores(sentence)
    if vs.get('compound') >= 0:
        pos_reviews += ' %s' % sentence
    else:
        neg_reviews += ' %s' % sentence
    score_arr.append(vs.get('compound'))

from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)
stopwords.add('app')


# optionally add: stopwords=STOPWORDS and change the arg below
def generate_wordcloud(text):
    wordcloud = WordCloud(relative_scaling=1.0,
                          scale=3,
                          stopwords=stopwords
                          ).generate(text)
    plt.figure(figsize=(20, 20))
    plt.imshow(wordcloud)
    plt.axis("off")
    st.pyplot()


if st.button("Get Word cloud for Negative reviews"):
    st.success("Generating A Negative WordCloud")
    generate_wordcloud(neg_reviews)

if st.button("Get Word cloud for Positive reviews"):
    st.success("Generating A Positive WordCloud")
    generate_wordcloud(pos_reviews)

st.sidebar.header("About App")
st.sidebar.info("A Twitter Sentiment analysis Project which will scrap twitter for the topic selected by the user. The extracted tweets will then be used to determine the Sentiments of those tweets. \
                    The different Visualizations will help us get a feel of the overall mood of the people on Twitter regarding the topic we select.")
st.sidebar.text("Built with Streamlit")

st.sidebar.header("For Any Queries/Suggestions Please reach out at :")
st.sidebar.info("darekarabhishek@gmail.com")

if st.button("Exit"):
    st.balloons()
