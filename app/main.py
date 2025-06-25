import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import joblib
import re
from streamlit_lottie import st_lottie
import requests

# --- App Configuration ---
st.set_page_config(
    page_title="🧠 MHEDA • Emotion Tracker",
    page_icon="🫂",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Load Resources ---
@st.cache_resource
def load_resources():
    model = joblib.load("models/emotion_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_resources()

# --- Lottie Animations ---
def load_lottie(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return None
    return None

# Using reliable LottieFiles URLs
lottie_celebrate = load_lottie("https://assets1.lottiefiles.com/packages/lf20_vyL7qy.json")
lottie_analytics = load_lottie("https://assets9.lottiefiles.com/packages/lf20_2glqweqs.json")

# --- Emotion Configuration ---
EMOTION_COLORS = {
    'joy': '#FFD166', 'love': '#EF476F', 'surprise': '#06D6A0',
    'gratitude': '#118AB2', 'admiration': '#073B4C', 'amusement': '#FF9E6D',
    'anger': '#FF6B6B', 'sadness': '#6A67CE', 'fear': '#4ECDC4',
    'neutral': '#B8B8B8'
}

label_map = {
    0: 'admiration', 1: 'amusement', 2: 'anger', 3: 'annoyance', 4: 'approval',
    5: 'caring', 6: 'confusion', 7: 'curiosity', 8: 'desire', 9: 'disappointment',
    10: 'disapproval', 11: 'disgust', 12: 'embarrassment', 13: 'excitement',
    14: 'fear', 15: 'gratitude', 16: 'grief', 17: 'joy', 18: 'love', 19: 'nervousness',
    20: 'neutral', 21: 'optimism', 22: 'pride', 23: 'realization', 24: 'relief',
    25: 'remorse', 26: 'sadness', 27: 'surprise'
}

# --- Text Processing ---
def clean_text(text):
    import nltk
    from nltk.corpus import stopwords
    
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        stop_words = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered)

# --- UI Components ---
def emotion_card(emotion):
    color = EMOTION_COLORS.get(emotion, '#073B4C')
    return f"""
        <div style="
            background: {color};
            border-radius: 16px;
            padding: 20px;
            color: white;
            text-align: center;
            margin: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            animation: pulse 2s infinite;
        ">
            <h2 style="margin:0; font-size: 2.5rem;">{emotion.upper()}</h2>
        </div>
    """

# --- App Layout ---
with st.container():
    col1, col2 = st.columns([1, 3])
    with col1:
        if lottie_celebrate:
            st_lottie(lottie_celebrate, height=120, key="header-animation")
        else:
            st.image("https://cdn-icons-png.flaticon.com/512/1995/1995485.png", width=100)
    with col2:
        st.title("🧠 MHEDA")
        st.markdown("### Mental Health Emotion Detection Assistant", unsafe_allow_html=True)
        st.caption("Track your feelings • Understand yourself • Grow emotionally")

st.divider()

# --- Main Input Section ---
with st.container():
    st.subheader("✨ Today's Journal")
    journal = st.text_area(
        "Share your thoughts...",
        height=200,
        placeholder="Dear diary... today I felt...",
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("💡 Tip: Be honest - this is your private space")
    with col2:
        analyze_btn = st.button("Analyze →", use_container_width=True)

# --- Processing & Results ---
if analyze_btn:
    if not journal.strip():
        st.warning("📣 Write something to analyze!")
    else:
        with st.spinner("🧠 Reading between the lines..."):
            cleaned = clean_text(journal)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            emotion = prediction

        # Store in session
        if 'history' not in st.session_state:
            st.session_state.history = []
        
        st.session_state.history.append({
            "date": datetime.now().date(),
            "entry": journal,
            "emotion": emotion
        })

        st.success("✅ Analysis complete!")
        st.balloons()
        
        # Emotion display
        st.markdown(emotion_card(emotion), unsafe_allow_html=True)
        
        # Personalized tips
        TIPS = {
            'sadness': "🎧 Try listening to uplifting music - sometimes vibrations heal better than words",
            'anger': "🥊 Do a 5-minute intense workout - channel that energy productively!",
            'fear': "📝 Name your fear and challenge its probability - most fears never happen",
            'joy': "🌟 Celebrate this moment! Text someone who amplifies your happiness",
            'neutral': "🌱 Try something new - novelty sparks emotional growth",
            'grief': "🕯️ Light a candle and honor your feelings - grief is love with nowhere to go",
            'disappointment': "🌦️ Remember: This feeling is temporary like weather - brighter days are coming",
            'excitement': "🎯 Channel this energy into a creative project!",
            'love': "💌 Send a heartfelt message to someone you cherish"
        }
        
        tip = TIPS.get(emotion, "💖 You're doing great just by checking in with yourself")
        with st.expander(f"💡 Personalized Tip for {emotion}"):
            st.info(tip)
            if emotion in ['sadness', 'grief', 'anger', 'despair']:
                st.link_button("📞 Crisis Resources", "https://www.thelivelovelaughfoundation.org/find-help/helplines")

# --- History Visualization ---
if 'history' in st.session_state and st.session_state.history:
    st.divider()
    st.subheader("📊 Your Emotional Journey")
    
    with st.expander("See Full History"):
        df = pd.DataFrame(st.session_state.history)
        df['date'] = pd.to_datetime(df['date'])
        
        # Interactive Plotly chart
        fig = px.scatter(
            df,
            x='date',
            y='emotion',
            color='emotion',
            color_discrete_map=EMOTION_COLORS,
            size=[40]*len(df),
            hover_name='emotion',
            hover_data={'date': True, 'emotion': False},
            template='plotly_white'
        )
        
        fig.update_layout(
            yaxis_title="",
            xaxis_title="",
            showlegend=False,
            height=400,
            plot_bgcolor='rgba(0,0,0,0.03)',
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
        st.plotly_chart(fig, use_container_width=True)
        
        # Emotion distribution pie chart
        st.subheader("😌 Your Emotional Palette")
        emotion_counts = df['emotion'].value_counts().reset_index()
        emotion_counts.columns = ['emotion', 'count']
        
        pie = px.pie(
            emotion_counts,
            names='emotion',
            values='count',
            color='emotion',
            color_discrete_map=EMOTION_COLORS,
            hole=0.4
        )
        pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(pie, use_container_width=True)
        
        # Raw data table
        st.dataframe(df.sort_values('date', ascending=False), hide_index=True)

# --- Sidebar & Footer ---
st.sidebar.header("About MHEDA")
st.sidebar.markdown("""
    Your personal emotion tracker that helps:
    - 🔍 Identify emotional patterns
    - 🌱 Grow emotional intelligence
    - 💖 Practice mindful self-reflection
""")

st.sidebar.divider()
st.sidebar.caption("Made with ❤️ in India")
st.sidebar.caption("v1.0 • [Privacy Policy](https://example.com)")

# --- CSS Styling ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@700&family=Nunito:wght@400;600;800&display=swap');
        
        body {
            font-family: 'Nunito', sans-serif;
            background-color: #fafafa;
        }
        
        h1, h2, h3 {
            font-family: 'Comfortaa', cursive;
            color: #5e17eb;
        }
        
        .stButton>button {
            background: linear-gradient(135deg, #6e8efb, #a777e3);
            border-radius: 12px;
            color: white;
            font-weight: 800;
            padding: 10px 24px;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(110, 142, 251, 0.25);
        }
        
        .stTextArea textarea {
            border-radius: 16px !important;
            padding: 16px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }
        
        .st-emotion-cache-1v0mbdj img {
            border-radius: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)