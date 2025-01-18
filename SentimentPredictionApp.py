import streamlit as st
import pandas as pd
import time
from textblob import TextBlob
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords only once at the start of the app
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

st.info("Stopwords have been downloaded. You may proceed with uploading your file.")

# Title
st.title("Sentiment Analysis App with TextBlob and Time Reporting")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    start_time = time.time()
    # Read the file
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())
    st.write(f"Step 1: Loaded CSV file - Time taken: {time.time() - start_time:.2f} seconds")

    if 'review' in df.columns and 'sentiment' in df.columns:
        step_time = time.time()
        reviews = df['review']

        # Preprocessing function
        def preprocess_text(text):
            text = text.lower()  # Lowercase
            text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
            words = text.split()
            words = [word for word in words if word not in stop_words]  # Remove stopwords
            return ' '.join(words)

        # Apply preprocessing
        st.write("Preprocessing text data...")
        df['cleaned_review'] = reviews.apply(preprocess_text)
        st.write(f"Step 2: Preprocessed text data - Time taken: {time.time() - step_time:.2f} seconds")

        step_time = time.time()
        # Sentiment analysis using TextBlob
        st.write("Performing sentiment analysis...")

        def get_sentiment(text):
            analysis = TextBlob(text)
            return "positive" if analysis.sentiment.polarity > 0 else "negative"

        df['predicted_sentiment'] = df['cleaned_review'].apply(get_sentiment)
        st.write(f"Step 3: Sentiment analysis completed - Time taken: {time.time() - step_time:.2f} seconds")

        # Display sentiment comparison
        correct_predictions = sum(df['predicted_sentiment'] == df['sentiment'])
        total_predictions = len(df)
        accuracy = correct_predictions / total_predictions

        st.write("### Sentiment Analysis Results")
        st.write(f"Total Predictions: {total_predictions}")
        st.write(f"Correct Predictions: {correct_predictions}")
        st.write(f"Accuracy: {accuracy:.2f}")

        total_time = time.time() - start_time
        st.write(f"Total time taken for all steps: {total_time:.2f} seconds")
    else:
        st.error("The CSV file must contain 'review' and 'sentiment' columns.")
