import streamlit as st
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import string
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Title
st.title("Sentiment Analysis App with Time Reporting")

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
        sentiments = df['sentiment']

        # Preprocessing function
        def preprocess_text(text):
            text = text.lower()  # Lowercase
            text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
            words = text.split()
            words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
            return ' '.join(words)

        # Apply preprocessing
        st.write("Preprocessing text data...")
        df['cleaned_review'] = reviews.apply(preprocess_text)
        st.write(f"Step 2: Preprocessed text data - Time taken: {time.time() - step_time:.2f} seconds")

        step_time = time.time()
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_review'], sentiments, test_size=0.2, random_state=42)
        st.write(f"Step 3: Split data into training and test sets - Time taken: {time.time() - step_time:.2f} seconds")

        step_time = time.time()
        # Convert text data to a matrix of token counts
        vectorizer = CountVectorizer()
        X_train_matrix = vectorizer.fit_transform(X_train)
        X_test_matrix = vectorizer.transform(X_test)
        st.write(f"Step 4: Converted text to matrix form - Time taken: {time.time() - step_time:.2f} seconds")

        step_time = time.time()
        # Logistic Regression model
        st.write("Building Logistic Regression model...")
        logistic_model = LogisticRegression()
        logistic_model.fit(X_train_matrix, y_train)
        y_train_pred_logistic = logistic_model.predict(X_train_matrix)
        y_test_pred_logistic = logistic_model.predict(X_test_matrix)

        logistic_train_accuracy = accuracy_score(y_train, y_train_pred_logistic)
        logistic_test_accuracy = accuracy_score(y_test, y_test_pred_logistic)
        st.write(f"Step 5: Trained Logistic Regression model - Time taken: {time.time() - step_time:.2f} seconds")

        step_time = time.time()
        # Decision Tree model
        st.write("Building Decision Tree model...")
        decision_tree_model = DecisionTreeClassifier()
        decision_tree_model.fit(X_train_matrix, y_train)
        y_train_pred_tree = decision_tree_model.predict(X_train_matrix)
        y_test_pred_tree = decision_tree_model.predict(X_test_matrix)

        tree_train_accuracy = accuracy_score(y_train, y_train_pred_tree)
        tree_test_accuracy = accuracy_score(y_test, y_test_pred_tree)
        st.write(f"Step 6: Trained Decision Tree model - Time taken: {time.time() - step_time:.2f} seconds")

        total_time = time.time() - start_time
        # Display accuracy results
        st.write("### Model Performance")
        st.write(f"Logistic Regression - Training Accuracy: {logistic_train_accuracy:.2f}")
        st.write(f"Logistic Regression - Test Accuracy: {logistic_test_accuracy:.2f}")
        st.write(f"Decision Tree - Training Accuracy: {tree_train_accuracy:.2f}")
        st.write(f"Decision Tree - Test Accuracy: {tree_test_accuracy:.2f}")
        st.write(f"Total time taken for all steps: {total_time:.2f} seconds")
    else:
        st.error("The CSV file must contain 'review' and 'sentiment' columns.")
