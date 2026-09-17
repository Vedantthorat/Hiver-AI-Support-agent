import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from src.semantic_intents import PrototypeIntentClassifier


st.set_page_config(
    page_title="Amazon Customer Support AI",
    page_icon="🤖",
    layout="centered"
)


@st.cache_resource
def load_intent_classifier():
    training_data = pd.read_csv(
        "data/processed/amazonhelp_training_135.csv"
    )

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    classifier = PrototypeIntentClassifier(
        embedding_model=embedding_model
    )

    classifier.train(
        texts=training_data["text"].fillna("").tolist(),
        labels=training_data["intent"].tolist()
    )

    return classifier


st.title("🤖 Amazon Customer Support AI Agent")

st.write(
    "Enter an Amazon customer-support message to classify "
    "its intent."
)

customer_message = st.text_area(
    "Customer message",
    placeholder="Example: My package has not arrived yet.",
    height=150
)

if st.button("Analyze Message"):

    if not customer_message.strip():
        st.warning("Please enter a customer message.")

    else:
        with st.spinner("Analyzing customer message..."):

            classifier = load_intent_classifier()

            predicted_intent = classifier.predict_one(
                customer_message
            )

        st.subheader("Predicted Intent")

        st.success(predicted_intent)