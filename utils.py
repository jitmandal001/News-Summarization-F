import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Dict, Any
import numpy as np
from transformers import pipeline
import urllib.parse
from sklearn.feature_extraction.text import TfidfVectorizer
import tldextract
from deep_translator import GoogleTranslator
from playsound import playsound
import soundfile as sf
from transformers import AutoModel, AutoTokenizer
def search_news(company_name: str, num_articles: int = 2) -> List[str]:
    search_url = f"https://www.google.com/search?q={company_name}+news&tbm=nws"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        article_links = []
        for article in soup.select('.SoaBEf'):
            link_element = article.select_one('a')
            if link_element and 'href' in link_element.attrs:
                href = link_element['href']
                if href.startswith('/url?q='):
                    url = href.split('/url?q=')[1].split('&')[0]
                    url = urllib.parse.unquote(url)
                    article_links.append(url)
                elif href.startswith('http'):
                    article_links.append(href)

                if len(article_links) >= num_articles:
                    break

        return article_links
    except Exception as e:
        print(f"Error fetching news articles: {e}")
        return []

def extract_article_content(url: str) -> Dict[str, Any]:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.find("h1").get_text().strip() if soup.find("h1") else "No title found"

        content_element = soup.find("article") or soup.find("main") or soup.find("div", class_=["content", "article", "story"])
        content = " ".join([p.get_text().strip() for p in content_element.find_all("p")]) if content_element else "No content found"

        date_element = soup.find("time")
        date = date_element["datetime"] if date_element and "datetime" in date_element.attrs else None

        return {
            'url': url,
            'title': title,
            'content': content,
            'date': date
        }
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return {
            'url': url,
            'title': "Error extracting content",
            'content': "Error extracting content",
            'date': None
        }

def get_company_news(company_name: str) -> List[Dict[str, Any]]:
    """
    Fetch exactly 10 news articles for a given company.
    If fewer than 10 articles are retrieved initially, retry fetching more.
    """
    max_articles = 10
    articles = []
    retries = 3  # Number of retries to fetch missing articles

    for attempt in range(retries):
        # Fetch article URLs
        article_urls = search_news(company_name, num_articles=max_articles - len(articles))

        # Process each URL to extract content
        for url in article_urls:
            try:
                article_data = extract_article_content(url)
                # Avoid duplicates by checking the URL
                if article_data['url'] not in [a['url'] for a in articles]:
                    articles.append(article_data)
            except Exception as e:
                print(f"Error extracting from {url}: {e}")

        # Break if we have enough articles
        if len(articles) >= max_articles:
            break

    # If still fewer than 10 articles, fill with placeholders
    while len(articles) < max_articles:
        articles.append({
            'url': 'N/A',
            'title': 'No Title Available',
            'content': 'No Content Available',
            'date': None
        })

    return articles
def summarize_article(content: str, max_length: int = 50) -> str:
    summarizer = pipeline("summarization")
    max_input_length = summarizer.model.config.max_position_embeddings  # Get model's max input length

    # Ensure content does not exceed max input length
    truncated_content = content[:max_input_length]

    summary = summarizer(truncated_content, max_length=max_length, min_length=0, do_sample=False)
    return summary[0]['summary_text']

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment of the given text.

    Args:
        text: The text to analyze.

    Returns:
        Dictionary containing sentiment category and score.
    """
    try:
        # Initialize sentiment analyzer
        sentiment_analyzer = pipeline("sentiment-analysis", truncation=True)

        # Truncate text manually to avoid exceeding token limits
        max_token_limit = 512  # Most transformer models have a 512-token limit
        words = text.split()
        if len(words) > max_token_limit:
            text = ' '.join(words[:max_token_limit])

        # Perform sentiment analysis
        result = sentiment_analyzer(text)

        # Determine sentiment category based on label and score
        sentiment_category = "Positive" if result[0]['label'] == "POSITIVE" else "Negative"
        score = result[0]['score']

        # Add neutral category for borderline cases
        if 0.4 <= score <= 0.6:
            sentiment_category = "Neutral"

        return {
            'sentiment': sentiment_category,
            'score': score
        }
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return {
            'sentiment': "Unknown",
            'score': 0.0
        }

def extract_key_topics(text: str, num_topics: int = 5) -> List[str]:
    if len(text.split()) < 10:
        return ["Not enough text to extract topics"]

    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    sorted_indices = np.argsort(tfidf_scores)[::-1]
    top_topics = [feature_names[idx] for idx in sorted_indices[:num_topics]]

    return top_topics

def perform_comparative_analysis(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    sentiment_counts = {
        'Positive': len([a for a in articles if a['sentiment']['sentiment'] == 'Positive']),
        'Neutral': len([a for a in articles if a['sentiment']['sentiment'] == 'Neutral']),
        'Negative': len([a for a in articles if a['sentiment']['sentiment'] == 'Negative'])
    }

    all_topics = [topic for article in articles for topic in article['topics']]
    topic_frequency = {}
    for topic in all_topics:
        topic_frequency[topic] = topic_frequency.get(topic, 0) + 1

    common_topics = sorted(topic_frequency.items(), key=lambda x: x[1], reverse=True)

    sentiment_by_source = {}
    for article in articles:
        source = extract_source_from_url(article['url'])
        if source not in sentiment_by_source:
            sentiment_by_source[source] = []
        sentiment_by_source[source].append(article['sentiment']['sentiment'])

    return {
        'sentiment_distribution': sentiment_counts,
        'common_topics': common_topics[:10],
        'sentiment_by_source': sentiment_by_source
    }

def extract_source_from_url(url: str) -> str:
    extracted_info = tldextract.extract(url)
    return extracted_info.domain

from typing import List, Dict, Any
from transformers import pipeline

def get_combined_summary(articles, max_length: int = 100) -> str:
    """
    Generate a combined summary from multiple news articles.

    Args:
        articles: List of article dictionaries containing content
        max_length: Maximum length of the final summary

    Returns:
        A comprehensive summary combining insights from all articles
    """
    # Combine all article contents with titles as context
    combined_content = ""
    for article in articles:
        # Use .get() with default values to handle missing keys
        title = article.get('title', 'No Title')
        content = article.get('content', 'Content not available')
        combined_content += f"Article: {title}\n{content}\n\n"

    # Initialize the summarizer
    summarizer = pipeline("summarization")

    # Handle token limit constraints
    max_input_length = summarizer.model.config.max_position_embeddings
    truncated_content = combined_content[:max_input_length]

    # Generate the combined summary
    summary = summarizer(truncated_content, max_length=max_length, min_length=30, do_sample=False)

    # Handle different return formats from the pipeline
    if isinstance(summary, list):
        return summary[0]['summary_text']
    else:
        return summary['summary_text']

def generate_hindi_summary(combined_summary: str) -> str:
    """
    Translate the combined summary to Hindi using deep-translator.

    Args:
        combined_summary: The English combined summary

    Returns:
        The Hindi translation of the combined summary
    """
    try:
        translator = GoogleTranslator(source='auto', target='hi')
        hindi_summary = translator.translate(text=combined_summary)
        return hindi_summary
    except Exception as e:
        print(f"Error in translation: {e}")
        return "Translation failed"
def generate_hindi_speech(hindi_summary: str):
    """
    Convert Hindi summary to speech using AI4Bharat's VITS-Rasa-13 model and play it

    Args:
        hindi_summary: Hindi text summary to synthesize (max 500 characters)
    """
    try:
        # Load pre-trained model (requires CUDA-enabled GPU)
        model = AutoModel.from_pretrained("ai4bharat/vits_rasa_13", trust_remote_code=True).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained("ai4bharat/vits_rasa_13", trust_remote_code=True)

        # Process text and generate speech
        inputs = tokenizer(text=hindi_summary, return_tensors="pt").to("cuda")

        # Use default Indian voice profile (speaker_id=16 for male, 17 for female)
        outputs = model(inputs['input_ids'], speaker_id=16, emotion_id=0)

        # Convert to numpy array and save as temporary file
        audio_data = outputs.waveform.squeeze().cpu().numpy()
        sf.write("temp_hindi_speech.wav", audio_data, model.config.sampling_rate)

        # Play the audio using playsound
        playsound("temp_hindi_speech.wav")

    except Exception as e:
        print(f"Error in speech generation or playback: {e}")
        