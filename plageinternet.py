import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import string

app = Flask(__name__)

# Download the stopwords from NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def scrape_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        return content
    except Exception as e:
        print("Error scraping content:", e)
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_plagiarism', methods=['POST'])
def check_plagiarism():
    data = request.get_json()
    doc1 = data['documents'][0]
    doc2 = data['documents'][1]

    # Scraping content from Wikipedia for the second document
    wiki_url = 'https://en.wikipedia.org/wiki/' + doc2.replace(' ', '_')
    doc2_content = scrape_content(wiki_url)

    if doc2_content:
        processed_doc1 = preprocess(doc1)
        processed_doc2 = preprocess(doc2_content)

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([processed_doc1, processed_doc2])
        similarity_matrix = cosine_similarity(tfidf_matrix)

        return jsonify(similarity_matrix.tolist())
    else:
        return jsonify({"error": "Failed to scrape content from Wikipedia."})

if __name__ == '__main__':
    app.run(debug=True)
