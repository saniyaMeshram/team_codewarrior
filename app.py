import streamlit as st
import pickle


model = pickle.load(open("fake_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown(
    """
    <style>
    body {
        background: linear-gradient(120deg, #1d2671, #c33764);
        color: white;
    }
    .stButton>button {
        background-color: #1d2671;
        color: white;
        font-size: 18px;
        padding: 10px 25px;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #c33764;
        color: white;
    }
    textarea {
        border-radius: 8px;
    }
    .stTextArea>div>div>textarea {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("📰 Fake News Detection System")
st.markdown("Detect whether a news article is **Fake** or **True** using Machine Learning.")


st.sidebar.header("Instructions")
st.sidebar.write("""
1. Paste a news article in the box below.
2. Click the **Predict** button.
3. The model will classify the news as Fake or True.
""")


news_text = st.text_area("Paste News Here:", height=200)


if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("⚠ Please enter some news text!")
    else:
        
        input_data = vectorizer.transform([news_text])
        prediction = model.predict(input_data)

        
        if prediction[0] == 0:
            st.error("🚨 Fake News Detected")
        else:
            st.success("✅ This News is True")


st.markdown(
    """
    <div style='text-align: center; margin-top: 50px; font-size: 14px;'>
    
    </div>
    """,
    unsafe_allow_html=True
)