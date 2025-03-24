from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from utils import (
    get_company_news,
    analyze_sentiment,
    summarize_article,
    extract_key_topics,
    perform_comparative_analysis,
    get_combined_summary,
    generate_hindi_summary,
    generate_hindi_speech,
)

app = FastAPI()

class CompanyRequest(BaseModel):
    company_name: str

class ArticleResponse(BaseModel):
    url: str
    title: str
    summary: str
    sentiment: Dict[str, Any]
    topics: List[str]

class AnalysisResponse(BaseModel):
    articles: List[ArticleResponse]
    comparative_analysis: Dict[str, Any]
    hindi_summary: str
    audio_path: str

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_company(request: CompanyRequest):
    try:
        # Get news articles
        raw_articles = get_company_news(request.company_name)
        
        if not raw_articles:
            raise HTTPException(status_code=404, detail="No articles found for this company")
        
        # Process each article
        processed_articles = []
        for article in raw_articles:
            try:
                # Use .get() to safely access keys
                summary = summarize_article(article.get('content', ''))
                sentiment = analyze_sentiment(article.get('content', ''))
                topics = extract_key_topics(article.get('content', ''))
                
                processed_articles.append({
                    'url': article.get('url', ''),
                    'title': article.get('title', 'No title'),
                    'summary': summary,
                    'sentiment': sentiment,
                    'topics': topics
                })
            except Exception as e:
                print(f"Error processing article: {e}")
                # Continue to next article instead of failing the entire process
                continue
        
        # Only continue if we have processed articles
        if not processed_articles:
            raise HTTPException(status_code=500, detail="Failed to process any articles")
        
        # Perform comparative analysis
        analysis = perform_comparative_analysis(processed_articles)
        
        # Generate combined summary
        combined_summary = get_combined_summary(processed_articles)
        
        # Generate Hindi summary
        hindi_summary = generate_hindi_summary(combined_summary)
        
        # Generate Hindi speech (audio file)
        generate_hindi_speech(hindi_summary)
        
        return {
            'articles': processed_articles,
            'comparative_analysis': analysis,
            'hindi_summary': hindi_summary,
            'audio_path': 'Audio playback handled directly'
        }
    
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions directly
        raise http_exc
    
    except Exception as e:
        # Log and convert other exceptions to HTTP exceptions
        print(f"Error in analyze_company: {e}")
        raise HTTPException(status_code=500, detail=str(e))