from flask import Flask, render_template
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

API_KEY = os.getenv("GNEWS_API_KEY")
BASE_URL = "https://gnews.io/api/v4/top-headlines"


@app.route("/")
def home():
    params = {
        "token": API_KEY,
        "lang": "en",
        "max": 5
    }
    response = requests.get(BASE_URL, params=params)

    print("STATUS:", response.status_code)  # Debug
    print("RESPONSE:", response.text)       # Debug

    if response.status_code == 200:
        news_data = response.json().get("articles", [])
    else:
        news_data = []

    return render_template("test.html", news=news_data)


if __name__ == "__main__":
    app.run(debug=True)
