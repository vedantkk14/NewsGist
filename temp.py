import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

def download_models():
    models = [
        {
            "name": "facebook/bart-large-cnn",
            "path": "./models2/bart-large-cnn",
            "type": "seq2seq"
        },
        {
            "name": "google/pegasus-xsum",
            "path": "./models/pegasus-xsum",
            "type": "seq2seq"
        },
        {
            "name": "cardiffnlp/twitter-roberta-base-sentiment",
            "path": "./models_sentiment/roberta-sentiment",
            "type": "classification"
        }
    ]

    for model in models:
        print(f"Downloading {model['name']} to {model['path']}...")
        os.makedirs(model['path'], exist_ok=True)
        
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model['name'])
        tokenizer.save_pretrained(model['path'])
        
        # Download model
        if model['type'] == "seq2seq":
            model_obj = AutoModelForSeq2SeqLM.from_pretrained(model['name'])
        else:
            model_obj = AutoModelForSequenceClassification.from_pretrained(model['name'])
            
        model_obj.save_pretrained(model['path'])
        print(f"Successfully saved {model['name']}")

if __name__ == "__main__":
    download_models()
