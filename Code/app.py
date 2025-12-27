import streamlit as st
import pandas as pd
import altair as alt


# Page Configuration
st.set_page_config(
    page_title="The Sentiment Scope",
    page_icon="🔍",
    layout="wide"
)

# 1. Load Data (Cached for performance)
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Datasets/my_data2 (1).csv')
    except FileNotFoundError:
        return None

    # Helper function to detect language
    def detect_language(text):
        if pd.isna(text): return 'Unknown'
        # Check for Arabic unicode range
        if any("\u0600" <= c <= "\u06FF" for c in str(text)):
            return 'Arabic'
        return 'English'

    # Apply language detection
    df['language'] = df['review_content'].apply(detect_language)
    
    # Calculate review length
    df['length'] = df['review_content'].str.len()
    
    # Map labels to text for better readability
    label_map = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    df['sentiment_label'] = df['label'].map(label_map)
    
    return df

df = load_data()

if df is None:
    st.error("File 'my_data2.csv' not found. Please place it in the same directory.")
    st.stop()

# 2. Sidebar Controls
st.sidebar.header("🎛️ Control Panel")

# Language Filter
lang_filter = st.sidebar.radio(
    "Select Language:",
    options=["All", "English", "Arabic"],
    index=0
)

# Sentiment Filter
selected_sentiments = st.sidebar.multiselect(
    "Filter by Sentiment:",
    options=['Positive', 'Neutral', 'Negative'],
    default=['Positive', 'Neutral', 'Negative']
)

# Text Search
search_query = st.sidebar.text_input("🔍 Search keywords (e.g., 'pizza', 'خدمة')")

# Apply Filters
filtered_df = df.copy()

if lang_filter != "All":
    filtered_df = filtered_df[filtered_df['language'] == lang_filter]

filtered_df = filtered_df[filtered_df['sentiment_label'].isin(selected_sentiments)]

if search_query:
    filtered_df = filtered_df[filtered_df['review_content'].str.contains(search_query, case=False, na=False)]

# 3. Main Dashboard UI
st.title("🔍 The Sentiment Scope")
st.markdown("### Analyze customer sentiment across languages")

# Top KPI Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Reviews", f"{len(filtered_df):,}")
with col2:
    positive_pct = (len(filtered_df[filtered_df['label'] == 1]) / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
with col3:
    avg_len = filtered_df['length'].mean() if len(filtered_df) > 0 else 0
    st.metric("Avg Review Length", f"{int(avg_len)} chars")
with col4:
    arabic_pct = (len(filtered_df[filtered_df['language'] == 'Arabic']) / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    st.metric("Arabic Content", f"{arabic_pct:.1f}%")

st.divider()

# Row 2: Charts
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("📊 Sentiment Distribution")
    chart_data = filtered_df['sentiment_label'].value_counts().reset_index()
    chart_data.columns = ['Sentiment', 'Count']
    
    bar_chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('Sentiment', sort=['Positive', 'Neutral', 'Negative']),
        y='Count',
        color=alt.Color('Sentiment', scale={'domain': ['Positive', 'Neutral', 'Negative'], 'range': ['#2ecc71', '#95a5a6', '#e74c3c']}),
        tooltip=['Sentiment', 'Count']
    ).properties(height=300)
    
    st.altair_chart(bar_chart, use_container_width=True)

with col_chart2:
    st.subheader("📏 Review Length Distribution")
    hist_chart = alt.Chart(filtered_df).mark_bar().encode(
        x=alt.X('length', bin=alt.Bin(maxbins=30), title='Character Count'),
        y='count()',
        color='sentiment_label'
    ).properties(height=300)
    
    st.altair_chart(hist_chart, use_container_width=True)

st.divider()

# Row 3: Data Explorer
st.subheader("📝 Review Explorer")

st.dataframe(
    filtered_df[['sentiment_label', 'language', 'review_content', 'length']].head(100),
    use_container_width=True
)


