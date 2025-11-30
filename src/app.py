import streamlit as st

# UI Layout - MUST be first Streamlit command
st.set_page_config(page_title="Construction Sentiment AI", layout="wide")

import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load NLTK data (ensure it's available)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Load Model and Vectorizer
@st.cache_resource
def load_model():
    try:
        with open('sentiment_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_model()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

def predict_sentiment(text):
    if model is None or vectorizer is None:
        return None
    cleaned_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([cleaned_text]).toarray()
    prediction = model.predict(vectorized_text)[0]
    return prediction

def generate_daily_summary(df, selected_date=None):
    """Generate AI-powered daily summary"""
    if selected_date is None:
        selected_date = df['Date'].max()
    
    day_data = df[df['Date'] == selected_date]
    
    if len(day_data) == 0:
        return "No data available for this date."
    
    total_notes = len(day_data)
    sentiment_counts = day_data['Sentiment'].value_counts()
    
    positive_count = sentiment_counts.get('Positive', 0)
    neutral_count = sentiment_counts.get('Neutral', 0)
    negative_count = sentiment_counts.get('Negative', 0)
    
    positive_pct = (positive_count / total_notes) * 100
    neutral_pct = (neutral_count / total_notes) * 100
    negative_pct = (negative_count / total_notes) * 100
    
    # Generate summary text
    summary = f"## 📅 Daily Summary for {selected_date}\n\n"
    summary += f"**Total Activities Logged:** {total_notes}\n\n"
    
    # Overall sentiment
    if positive_pct > 60:
        summary += f"✅ **Overall Status:** Excellent progress! {positive_pct:.1f}% of activities were positive.\n\n"
    elif positive_pct > 40:
        summary += f"⚠️ **Overall Status:** Moderate progress with {positive_pct:.1f}% positive activities.\n\n"
    else:
        summary += f"🚨 **Overall Status:** Challenging day with only {positive_pct:.1f}% positive activities.\n\n"
    
    # Breakdown
    summary += f"**Sentiment Breakdown:**\n"
    summary += f"- 😊 Positive: {positive_count} ({positive_pct:.1f}%)\n"
    summary += f"- 😐 Neutral: {neutral_count} ({neutral_pct:.1f}%)\n"
    summary += f"- 😡 Negative: {negative_count} ({negative_pct:.1f}%)\n\n"
    
    # Key highlights
    if positive_count > 0:
        summary += f"**✨ Highlights:**\n"
        positive_notes = day_data[day_data['Sentiment'] == 'Positive']['Note'].head(3).tolist()
        for note in positive_notes:
            summary += f"- {note}\n"
        summary += "\n"
    
    # Issues to address
    if negative_count > 0:
        summary += f"**⚠️ Issues Requiring Attention:**\n"
        negative_notes = day_data[day_data['Sentiment'] == 'Negative']['Note'].head(3).tolist()
        for note in negative_notes:
            summary += f"- {note}\n"
        summary += "\n"
    
    # Recommendations
    if negative_pct > 30:
        summary += "**💡 Recommendations:**\n"
        summary += "- Schedule a team meeting to address recurring issues\n"
        summary += "- Review resource allocation and availability\n"
        summary += "- Consider contingency planning for identified risks\n"
    
    return summary

# Main UI
st.title("🏗️ Construction Site Sentiment Analysis")
st.markdown("Analyze job notes to detect site mood and potential issues early.")

# Sidebar for Navigation
page = st.sidebar.selectbox("Choose a page", ["Sentiment Analyzer", "Analytics Dashboard", "Daily Summary", "Upload Data"])

# Load or upload data
uploaded_file = st.sidebar.file_uploader("📁 Upload your CSV file (optional)", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Custom file loaded!")
    
    # Predict sentiment for uploaded data if not present
    if 'Sentiment' not in df.columns and 'Note' in df.columns:
        with st.sidebar:
            with st.spinner("Analyzing sentiments..."):
                df['Sentiment'] = df['Note'].apply(predict_sentiment)
        st.sidebar.success("✅ Sentiments predicted!")
else:
    try:
        df = pd.read_csv('construction_notes.csv')
    except FileNotFoundError:
        df = None

# PAGE 1: Sentiment Analyzer
if page == "Sentiment Analyzer":
    st.header("📝 Analyze a Job Note")
    
    if model is None:
        st.error("Model not found! Please run model.py first to train the model.")
    else:
        user_input = st.text_area("Enter a daily job note:", "e.g., Delay due to shortage of cement.", height=100)
        
        if st.button("Analyze Sentiment", type="primary"):
            if user_input:
                prediction = predict_sentiment(user_input)
                
                st.subheader("Result:")
                if prediction == "Positive":
                    st.success(f"😊 Sentiment: **{prediction}**")
                elif prediction == "Negative":
                    st.error(f"😡 Sentiment: **{prediction}**")
                else:
                    st.warning(f"😐 Sentiment: **{prediction}**")
            else:
                st.warning("Please enter some text.")

# PAGE 2: Analytics Dashboard
elif page == "Analytics Dashboard":
    st.header("📊 Advanced Analytics Dashboard")
    
    if df is None:
        st.error("No data available. Please upload a CSV file or run generate_data.py.")
    else:
        # Ensure Date column exists
        if 'Date' not in df.columns:
            st.warning("Date column not found. Adding current date to all entries.")
            df['Date'] = datetime.now().strftime('%Y-%m-%d')
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        total_notes = len(df)
        positive_count = len(df[df['Sentiment'] == 'Positive'])
        negative_count = len(df[df['Sentiment'] == 'Negative'])
        neutral_count = len(df[df['Sentiment'] == 'Neutral'])
        
        with col1:
            st.metric("📝 Total Notes", total_notes)
        with col2:
            st.metric("😊 Positive", f"{positive_count} ({positive_count/total_notes*100:.1f}%)")
        with col3:
            st.metric("😡 Negative", f"{negative_count} ({negative_count/total_notes*100:.1f}%)")
        with col4:
            st.metric("😐 Neutral", f"{neutral_count} ({neutral_count/total_notes*100:.1f}%)")
        
        st.divider()
        
        # Date Range Filter
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", df['Date'].min())
        with col2:
            end_date = st.date_input("End Date", df['Date'].max())
        
        # Filter data
        mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
        filtered_df = df[mask]
        
        # Row 1: Sentiment Trends Over Time
        st.subheader("📈 Sentiment Trends Over Time")
        daily_sentiment = filtered_df.groupby(['Date', 'Sentiment']).size().reset_index(name='Count')
        
        fig_trend = px.line(daily_sentiment, x='Date', y='Count', color='Sentiment',
                           color_discrete_map={'Positive': '#00CC96', 'Neutral': '#FFA15A', 'Negative': '#EF553B'},
                           title='Daily Sentiment Trends')
        fig_trend.update_layout(height=400)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Row 2: Two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🥧 Sentiment Distribution")
            sentiment_counts = filtered_df['Sentiment'].value_counts()
            fig_pie = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                            color=sentiment_counts.index,
                            color_discrete_map={'Positive': '#00CC96', 'Neutral': '#FFA15A', 'Negative': '#EF553B'})
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("📊 Sentiment by Category")
            fig_bar = px.bar(sentiment_counts, x=sentiment_counts.index, y=sentiment_counts.values,
                            color=sentiment_counts.index,
                            color_discrete_map={'Positive': '#00CC96', 'Neutral': '#FFA15A', 'Negative': '#EF553B'},
                            labels={'x': 'Sentiment', 'y': 'Count'})
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Row 3: Heatmap
        st.subheader("🗓️ Daily Sentiment Heatmap")
        
        # Create pivot table for heatmap
        filtered_df['Day'] = filtered_df['Date'].dt.day_name()
        filtered_df['Week'] = filtered_df['Date'].dt.isocalendar().week
        
        # Calculate sentiment score: Positive=1, Neutral=0, Negative=-1
        sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
        filtered_df['Sentiment_Score'] = filtered_df['Sentiment'].map(sentiment_map)
        
        heatmap_data = filtered_df.groupby(filtered_df['Date'].dt.date)['Sentiment_Score'].mean().reset_index()
        heatmap_data.columns = ['Date', 'Average_Sentiment']
        
        fig_heatmap = px.density_heatmap(filtered_df, x=filtered_df['Date'].dt.date, y='Sentiment',
                                         color_continuous_scale='RdYlGn',
                                         title='Sentiment Distribution Across Days')
        fig_heatmap.update_layout(height=300)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Word Cloud Section
        st.subheader("☁️ Word Clouds by Sentiment")
        sentiment_type = st.selectbox("Select Sentiment", ["Positive", "Negative", "Neutral"])
        
        subset = filtered_df[filtered_df['Sentiment'] == sentiment_type]
        if len(subset) > 0:
            text = " ".join(subset['Note'])
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            
            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis("off")
            st.pyplot(fig_wc)
        else:
            st.info(f"No {sentiment_type} notes found in the selected date range.")

# PAGE 3: Daily Summary
elif page == "Daily Summary":
    st.header("📋 Daily Work Summary")
    
    if df is None:
        st.error("No data available. Please upload a CSV file or run generate_data.py.")
    else:
        if 'Date' not in df.columns:
            st.warning("Date column not found. Adding current date to all entries.")
            df['Date'] = datetime.now().strftime('%Y-%m-%d')
        
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        
        # Date selector
        available_dates = sorted(df['Date'].unique(), reverse=True)
        selected_date = st.selectbox("Select Date", available_dates)
        
        # Generate and display summary
        summary = generate_daily_summary(df, selected_date)
        st.markdown(summary)
        
        # Show detailed notes
        with st.expander("📝 View All Notes for This Day"):
            day_notes = df[df['Date'] == selected_date][['Note', 'Sentiment']]
            st.dataframe(day_notes, use_container_width=True)
        
        # Export button
        if st.button("📥 Export Summary as Text"):
            st.download_button(
                label="Download Summary",
                data=summary,
                file_name=f"daily_summary_{selected_date}.txt",
                mime="text/plain"
            )

# PAGE 4: Upload Data
elif page == "Upload Data":
    st.header("📁 Upload Your Data")
    
    st.markdown("""
    ### Instructions
    Upload a CSV file with your construction job notes. The file should have the following format:
    
    | Column | Description | Required |
    |--------|-------------|----------|
    | Date | Date of the note (YYYY-MM-DD) | Optional |
    | Note | The job note text | Required |
    | Sentiment | Positive/Neutral/Negative | Optional |
    
    If the Sentiment column is missing, the system will automatically predict it for you.
    """)
    
    uploaded_file_page = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file_page is not None:
        try:
            df_upload = pd.read_csv(uploaded_file_page)
            st.success("✅ File uploaded successfully!")
            
            st.subheader("Preview")
            st.dataframe(df_upload.head(10))
            
            # Validate columns
            if 'Note' not in df_upload.columns:
                st.error("❌ Error: 'Note' column is required!")
            else:
                # Add predictions if needed
                if 'Sentiment' not in df_upload.columns:
                    if model is not None:
                        with st.spinner("Analyzing sentiments..."):
                            df_upload['Sentiment'] = df_upload['Note'].apply(predict_sentiment)
                        st.success("✅ Sentiments predicted!")
                
                # Add date if missing
                if 'Date' not in df_upload.columns:
                    df_upload['Date'] = datetime.now().strftime('%Y-%m-%d')
                    st.info("ℹ️ Date column added with current date.")
                
                # Show processed data
                st.subheader("Processed Data")
                st.dataframe(df_upload.head(10))
                
                # Download processed file
                csv = df_upload.to_csv(index=False)
                st.download_button(
                    label="📥 Download Processed CSV",
                    data=csv,
                    file_name="processed_construction_notes.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    # Sample data download
    st.divider()
    st.subheader("📥 Download Sample Template")
    sample_data = pd.DataFrame({
        'Date': ['2024-11-30', '2024-11-30', '2024-11-29'],
        'Note': ['Work completed on time.', 'Delay due to rain.', 'Inspection scheduled.'],
        'Sentiment': ['Positive', 'Negative', 'Neutral']
    })
    
    st.dataframe(sample_data)
    csv_sample = sample_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Sample CSV",
        data=csv_sample,
        file_name="sample_construction_notes.csv",
        mime="text/csv"
    )
