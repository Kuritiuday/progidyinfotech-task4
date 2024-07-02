

import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
file_path = 'twitter_training.csv'
df = pd.read_csv(file_path)

# Handle missing values by filling them with an empty string
df.iloc[:, -1] = df.iloc[:, -1].fillna('')

# Perform sentiment analysis using TextBlob
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Apply sentiment analysis to the cleaned tweet text
df['Sentiment'] = df.iloc[:, -1].apply(get_sentiment)

# Create visualizations
plt.figure(figsize=(10, 6))

# Distribution of sentiment polarity
sns.histplot(df['Sentiment'], bins=30, kde=True)
plt.title('Distribution of Sentiment Polarity')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.show()

# Sentiment analysis for specific topics or brands
# Assuming the second column contains the topic/brand
plt.figure(figsize=(12, 8))
sns.boxplot(x=df.iloc[:, 1], y=df['Sentiment'])
plt.title('Sentiment Polarity by Topic/Brand')
plt.xlabel('Topic/Brand')
plt.ylabel('Sentiment Polarity')
plt.xticks(rotation=90)
plt.show()

