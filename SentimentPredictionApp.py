import streamlit as st
import pandas as pd
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Title
st.title("Movie Data Sentiment Analysis with TensorFlow")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    start_time = time.time()
    # Read the file
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())
    st.write(f"Step 1: Loaded CSV file - Time taken: {time.time() - start_time:.2f} seconds")

    if 'Description' in df.columns:
        step_time = time.time()
        descriptions = df['Description']
        
        # Preprocessing function
        def preprocess_text(text):
            text = text.lower()  # Lowercase
            text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
            words = text.split()
            words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
            return ' '.join(words)

        # Apply preprocessing
        st.write("Preprocessing text data...")
        df['cleaned_description'] = descriptions.apply(preprocess_text)
        st.write(f"Step 2: Preprocessed text data - Time taken: {time.time() - step_time:.2f} seconds")

        step_time = time.time()
        # Tokenizing and padding sequences
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(df['cleaned_description'])
        sequences = tokenizer.texts_to_sequences(df['cleaned_description'])
        padded_sequences = pad_sequences(sequences, maxlen=200)
        st.write(f"Step 3: Tokenized and padded text sequences - Time taken: {time.time() - step_time:.2f} seconds")

        # Generate labels for sentiment (assuming binary sentiment for simplicity)
        df['sentiment'] = (df['Rating'] >= 6.0).astype(int)  # Assuming a rating of 6+ is positive sentiment

        X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['sentiment'], test_size=0.2, random_state=42)

        step_time = time.time()
        # Build TensorFlow model
        model = Sequential([
            Embedding(input_dim=5000, output_dim=64, input_length=200),
            LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        st.write("Step 4: Building and training the model...")
        model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1)
        st.write(f"Step 5: Trained TensorFlow model - Time taken: {time.time() - step_time:.2f} seconds")

        step_time = time.time()
        y_train_pred = (model.predict(X_train) > 0.5).astype(int)
        y_test_pred = (model.predict(X_test) > 0.5).astype(int)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        st.write(f"Step 6: Evaluated model performance - Time taken: {time.time() - step_time:.2f} seconds")

        total_time = time.time() - start_time
        # Display accuracy results
        st.write("### Model Performance")
        st.write(f"Training Accuracy: {train_accuracy:.2f}")
        st.write(f"Test Accuracy: {test_accuracy:.2f}")
        st.write(f"Total time taken for all steps: {total_time:.2f} seconds")

    else:
        st.error("The CSV file must contain a 'Description' column.")
