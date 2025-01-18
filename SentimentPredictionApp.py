import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Streamlit App Title
st.title("Sentiment Analysis App")
st.write("This app uses Logistic Regression and Decision Tree to predict the sentiment of user reviews.")

# Input data
st.sidebar.header("Input Data")
data_input = st.sidebar.text_area("Enter reviews (one per line):")

# Label input
labels_input = st.sidebar.text_area("Enter labels (0 for negative, 1 for positive) matching each review, comma-separated:")

# Process input data
if st.sidebar.button("Run Sentiment Analysis"):
    if data_input and labels_input:
        reviews = data_input.strip().split('\n')
        try:
            labels = list(map(int, labels_input.split(',')))
        except ValueError:
            st.error("Please ensure labels are integers (0 or 1) separated by commas.")
            st.stop()

        if len(reviews) != len(labels):
            st.error("The number of reviews and labels must match.")
            st.stop()

        # Data Preprocessing
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(reviews)
        y = labels

        # Splitting the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Logistic Regression Model
        lr_model = LogisticRegression()
        lr_model.fit(X_train, y_train)
        lr_predictions = lr_model.predict(X_test)
        lr_accuracy = accuracy_score(y_test, lr_predictions)
        st.subheader("Logistic Regression Model")
        st.write(f"Accuracy: {lr_accuracy:.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, lr_predictions))

        # Decision Tree Model
        dt_model = DecisionTreeClassifier()
        dt_model.fit(X_train, y_train)
        dt_predictions = dt_model.predict(X_test)
        dt_accuracy = accuracy_score(y_test, dt_predictions)
        st.subheader("Decision Tree Model")
        st.write(f"Accuracy: {dt_accuracy:.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, dt_predictions))

    else:
        st.error("Please enter both reviews and labels.")
