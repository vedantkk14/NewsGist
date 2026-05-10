import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
# REMOVING 'models/'
API_URL = "https://api-inference.huggingface.co/summarization/facebook/bart-large-cnn"
headers = {
	"Authorization": f"Bearer {HF_API_KEY}",
	"Content-Type": "application/json"
}

payload = {
	"inputs": "The tower is 324 metres tall.",
	"parameters": {
		"min_length": 5,
		"max_length": 20
	}
}

response = requests.post(API_URL, headers=headers, json=payload)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")
