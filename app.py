import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="News Sentiment Analyzer",
    page_icon="📰",
    layout="wide"
)

# Define API endpoint
API_ENDPOINT = "http://localhost:8000/analyze"

# Title and description
st.title("Company News Sentiment Analyzer")
st.markdown("""
This application extracts news articles about a company, performs sentiment analysis,
provides comparative insights, and generates a Hindi text-to-speech summary.
""")

# Company selection
company_name = st.text_input("Enter company name:", "").strip()
analyze_disabled = not bool(company_name)

if st.button("Analyze", disabled=analyze_disabled):
    if not company_name:
        st.warning("Please enter a valid company name")
        st.stop()

    with st.spinner("Analyzing news articles. This may take a few minutes..."):
        try:
            # Call API with timeout
            response = requests.post(
                API_ENDPOINT,
                json={"company_name": company_name},
                timeout=300  #5min timeout
            )

            if response.status_code == 200:
                data = response.json()

                if not data.get('articles', []):
                    st.warning("No articles found for this company. Try a different company name.")
                    st.stop()

                # Display articles in tabs
                st.subheader("News Articles Analysis")
                tabs = st.tabs([f"Article {i+1}" for i in range(len(data['articles']))])
                for i, (tab, article) in enumerate(zip(tabs, data['articles'])):
                    with tab:
                        st.markdown(f"### {article.get('title', 'No Title')}")
                        st.markdown(f"**Source:** [{article.get('url', '')}]({article.get('url', '')})")
                        st.markdown(f"**Summary:** {article.get('summary', 'No summary available')}")

                        # Display sentiment with color
                        sentiment = article.get('sentiment', {}).get('sentiment', 'Unknown')
                        sentiment_color = {
                            'Positive': '#4CAF50',  # Green
                            'Neutral': '#2196F3',  # Blue
                            'Negative': '#F44336'  # Red
                        }.get(sentiment, '#9E9E9E')  # Gray as default

                        st.markdown(
                            f"<span style='color:{sentiment_color};'>**Sentiment:** {sentiment} "
                            f"(Confidence: {article.get('sentiment', {}).get('score', 0):.2f})</span>",
                            unsafe_allow_html=True
                        )

                        # Display topics
                        topics = article.get('topics', [])
                        st.markdown(f"**Key Topics:** {', '.join(topics) if topics else 'No topics available'}")

                # Display comparative analysis
                st.subheader("Comparative Analysis")
                col1, col2 = st.columns([2, 3])

                with col1:
                    # Sentiment distribution chart
                    st.markdown("### Sentiment Distribution")
                    sentiment_dist = data['comparative_analysis']['sentiment_distribution']
                    fig, ax = plt.subplots()
                    ax.pie(
                        sentiment_dist.values(),
                        labels=sentiment_dist.keys(),
                        colors=['#4CAF50', '#2196F3', '#F44336'],
                        autopct='%1.1f%%',
                        startangle=90
                    )
                    ax.axis('equal')  # Equal aspect ratio ensures pie is drawn as circle
                    st.pyplot(fig)

                with col2:
                    # Common topics and sentiment by source
                    tab1, tab2 = st.tabs(["Common Topics", "Sentiment by Source"])

                    with tab1:
                        st.markdown("### Most Frequent Topics")
                        topics_df = pd.DataFrame(
                            data['comparative_analysis']['common_topics'],
                            columns=['Topic', 'Frequency']
                        )
                        st.dataframe(
                            topics_df.head(10),
                            use_container_width=True,
                            height=400
                        )

                    with tab2:
                        st.markdown("### Sentiment by News Source")
                        sentiment_by_source = data['comparative_analysis']['sentiment_by_source']
                        for source, sentiments in sentiment_by_source.items():
                            with st.expander(f"{source} ({len(sentiments)} articles)"):
                                sentiment_counts = pd.Series(sentiments).value_counts().to_dict()
                                st.bar_chart(sentiment_counts)

                # Hindi summary and audio section
                st.subheader("Hindi Summary")
                with st.container():
                    col_summary, col_audio = st.columns([3, 1])
                    with col_summary:
                        hindi_summary = data.get('hindi_summary', 'No Hindi summary available')
                        st.markdown(hindi_summary)
                    # with col_audio:
                        st.divider()
                        st.subheader("Audio Summary")
                        audio_path = data.get('audio_path', None)
                        if audio_path:
                            # st.markdown("### Audio Summary")
                            with open(audio_path, "rb") as audio_file:
                                audio_data = audio_file.read()
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                st.audio(audio_data)
                                # Download button for audio file
                                st.download_button(
                                    label="📥 Download Audio",
                                    data=audio_data,
                                    file_name=f"{company_name}_summary_{timestamp}.wav",
                                    mime="audio/wav"
                                )

                # Download CSV analysis report
                csv_data = pd.DataFrame(data['articles']).to_csv().encode('utf-8')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="📥 Download Analysis (CSV)",
                    data=csv_data,
                    file_name=f"{company_name}_analysis_{timestamp}.csv",
                    mime="text/csv"
                )

            else:
                # Handle API errors gracefully
                try:
                    error_detail = response.json().get('detail', response.text)
                except Exception:
                    error_detail = response.text

                st.error(f"API Error ({response.status_code}): {error_detail}")

        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the API. Please make sure the API server is running on http://localhost:8000")
        except requests.exceptions.Timeout:
            st.error("Request timed out. Please try again later.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
