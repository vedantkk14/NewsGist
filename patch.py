import sys
import os

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove transformers
content = content.replace('from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification', '# from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification')

# 2. Replace ModelManager
old_manager = content[content.find('# Global variables for model components'):content.find('# News API caching')]
new_manager = '''# Hugging Face API Configuration
HF_API_KEY = os.environ.get("HF_API_KEY", "") # Add your HF token here or in env vars
HF_API_URL_BART = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HF_API_URL_PEGASUS = "https://api-inference.huggingface.co/models/google/pegasus-xsum"
HF_API_URL_ROBERTA = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"

def query_hf_api(api_url, payload):
    """Query Hugging Face Inference API"""
    if not HF_API_KEY:
        return None, "Hugging Face API key not configured. Please set HF_API_KEY in app.py or environment variables."
        
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        if response.status_code == 503:
            return None, "Model is currently loading on Hugging Face servers. Please try again in about 20 seconds."
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.RequestException as e:
        logger.error(f"HF API Error: {str(e)}")
        if hasattr(e.response, 'text'):
            logger.error(f"Response: {e.response.text}")
        return None, f"API error: {str(e)}"

'''
content = content.replace(old_manager, new_manager)

# 3. Replace cached_summarize
old_cached = content[content.find('@lru_cache(maxsize=100)'):content.find('def fetch_gnews_data(endpoint, params):')]
new_cached = '''@lru_cache(maxsize=100)
def cached_summarize(text_hash, original_text, summary_type="detailed", min_len=10, max_len=50):
    """Cache summaries to avoid re-processing identical text"""
    if summary_type == "oneliner":
        api_url = HF_API_URL_PEGASUS
    else:
        api_url = HF_API_URL_BART
    
    try:
        payload = {
            "inputs": original_text,
            "parameters": {
                "min_length": min_len,
                "max_length": max_len,
                "do_sample": False,
                "truncation": True
            }
        }
        result, error = query_hf_api(api_url, payload)
        if error or not isinstance(result, list) or len(result) == 0 or "summary_text" not in result[0]:
            return None
        return result[0]["summary_text"]
    except Exception as e:
        logger.error(f"Error during cached summarization API call: {str(e)}")
        return None

'''
content = content.replace(old_cached, new_cached)

# 4. Replace generate_summary
old_generate = content[content.find('def generate_summary(text, summary_type="detailed", min_length=None, max_length=None):'):content.find('def analyze_sentiment(text):')]
new_generate = '''def generate_summary(text, summary_type="detailed", min_length=None, max_length=None):
    """Generate summary based on type with appropriate parameters"""
    try:
        # Validate and adjust parameters
        min_len = min_length or (10 if summary_type == "oneliner" else 30)
        max_len = max_length or (40 if summary_type == "oneliner" else 150)
        min_len, max_len = validate_length_params(min_len, max_len, text)
        
        start_time = time.time()
        
        text_word_count = len(text.split())
        if summary_type == "oneliner":
            max_len = min(max_len, max(15, text_word_count // 8))
            min_len = min(min_len, max(5, max_len - 5))
        else:
            if text_word_count > 500:
                max_len = min(max_len + 50, 250)
                min_len = min(min_len + 20, max_len - 10)
        
        if min_len >= max_len:
            min_len = max(1, max_len - 5)
        
        # Use caching for performance
        text_hash = get_text_hash(text)
        cached_result = cached_summarize(text_hash, text, summary_type, min_len, max_len)
        
        processing_time = time.time() - start_time
        
        if cached_result:
            return {
                "summary": cached_result,
                "type": summary_type,
                "original_length": len(text),
                "original_word_count": text_word_count,
                "summary_length": len(cached_result),
                "summary_word_count": len(cached_result.split()),
                "compression_ratio": round(len(cached_result) / len(text), 3),
                "processing_time": round(processing_time, 2),
                "model_used": "Pegasus-XSUM (API)" if summary_type == "oneliner" else "BART-Large-CNN (API)",
                "cached": True 
            }, None
            
        return None, "Error generating summary (API returned empty or failed). Please check your API token."
        
    except Exception as e:
        logger.error(f"Error generating {summary_type} summary: {str(e)}")
        return None, f"Error generating {summary_type} summary: {str(e)}"

'''
content = content.replace(old_generate, new_generate)

# 5. Replace analyze_sentiment
old_sentiment = content[content.find('def analyze_sentiment(text):'):content.find('@app.route("/")')]
new_sentiment = '''def analyze_sentiment(text):
    """Analyze sentiment of given text"""
    try:
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]
            logger.info(f"Text truncated to {max_length} characters for sentiment analysis")
            
        payload = {"inputs": text}
        result, error = query_hf_api(HF_API_URL_ROBERTA, payload)
        
        if error:
            return None, error
            
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
            # API returns a list of lists of dicts
            sentiment_results = sorted(result[0], key=lambda x: x['score'], reverse=True)
            best_match = sentiment_results[0]
            sentiment_label = best_match['label'].upper()
            confidence = best_match['score']
            all_results = result[0]
        elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict) and "label" in result[0]:
            # Sometimes API returns list of dicts directly
            best_match = result[0]
            sentiment_label = best_match['label'].upper()
            confidence = best_match['score']
            all_results = result
        else:
            return None, f"Unexpected API response: {str(result)}"
        
        label_mapping = {
            'LABEL_0': 'Negative',
            'LABEL_1': 'Neutral', 
            'LABEL_2': 'Positive',
            'NEGATIVE': 'Negative',
            'NEUTRAL': 'Neutral',
            'POSITIVE': 'Positive'
        }
        
        sentiment_name = label_mapping.get(sentiment_label, sentiment_label)
        
        return {
            'sentiment': sentiment_name,
            'confidence': round(confidence, 4),
            'raw_label': sentiment_label,
            'text_analyzed_length': len(text),
            'all_results': all_results
        }, None
        
    except Exception as e:
        logger.error(f"Error during sentiment analysis: {str(e)}")
        return None, f"Error analyzing sentiment: {str(e)}"

'''
content = content.replace(old_sentiment, new_sentiment)

# 6. Replace health & stats
old_health = content[content.find('@app.route("/health")'):content.find('@app.route(\'/summarize_pegasus\', methods=[\'POST\'])\n')]
new_health = '''@app.route("/health")
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "api_mode": True,
        "api_key_configured": bool(HF_API_KEY),
        "news_cache_size": len(news_cache),
        "timestamp": time.time()
    })

@app.route("/stats")
def get_stats():
    """Get system statistics"""
    cache_info = {}
    if hasattr(cached_summarize, 'cache_info'):
        info = cached_summarize.cache_info()
        cache_info = {
            "hits": info.hits,
            "misses": info.misses,
            "size": info.currsize,
            "maxsize": info.maxsize,
            "hit_rate": round(info.hits / (info.hits + info.misses) * 100, 2) if (info.hits + info.misses) > 0 else 0
        }
    
    return jsonify({
        "mode": "api",
        "api_key_configured": bool(HF_API_KEY),
        "cache_info": cache_info,
        "news_cache_entries": len(news_cache)
    })

'''
content = content.replace(old_health, new_health)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('app.py successfully patched for API use!')
