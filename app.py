import streamlit as st
import pickle


model = pickle.load(open("fake_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection System")
st.write("Enter a news article below and the model will predict if it is Fake or True.")

news_text = st.text_area("Paste News Content Here")

if st.button("Predict"):

    if news_text.strip() == "":
        st.warning("⚠ Please enter some news text.")
    else:
        input_data = vectorizer.transform([news_text])
        prediction = model.predict(input_data)

        if prediction[0] == 0:
            st.error("🚨 Fake News Detected")
        else:
            st.success("✅ This News is True")