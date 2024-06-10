import nltk
import requests
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# Download necessary NLTK data
nltk.download('all')

#initialise analyser
analyzer = SentimentIntensityAnalyzer()

# Initialize DataFrame
df = pd.DataFrame(columns=['reviewText'])

# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return ' '.join(lemmatized_tokens)

# Fetch Reddit data with error handling and delay
def fetch_reddit_data(url, headers, retries=1):
    for i in range(retries):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                wait_time = 2 ** i
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"HTTP error occurred: {e}")
                break
        except requests.exceptions.RequestException as e:
            print(f"Request error occurred: {e}")
            break
    return None

# Fetch items data
def fetch_items_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching items data: {e}")
        return None

# Extract item titles
def extract_item_titles(html):
    items = []
    for i in range(len(html)):
        if html[i:i+10] == "item-title":
            s = ""
            j = 12
            while html[i+j] != '<':
                s += html[i+j]
                j += 1
            items.append(s)
    return items

# Process Reddit comments
def process_reddit_comments(data, headers):
    global df
    while True:
        try:
            for post in data["data"]["children"]:
                title = post["data"]["title"]
                print(title)
                df = pd.concat([df, pd.DataFrame({'reviewText': [title]})], ignore_index=True)
                
                permalink = post["data"]["permalink"]
                comments_url = f"https://www.reddit.com{permalink[:-1]}.json"
                comments_data = fetch_reddit_data(comments_url, headers)
                
                if comments_data:
                    try:
                        for comment in comments_data[1]["data"]["children"]:
                            if "body" in comment["data"]:
                                body = comment["data"]["body"]
                                print(body)
                                df = pd.concat([df, pd.DataFrame({'reviewText': [body]})], ignore_index=True)
                    except (IndexError, KeyError) as e:
                        print(f"Error processing comments: {e}")
                        continue
            
            after = data["data"].get("after")
            if not after:
                break
            
            next_page_url = f"https://www.reddit.com/r/bindingofisaac.json?after={after}"
            data = fetch_reddit_data(next_page_url, headers)
            if not data:
                break
        except Exception as e:
            print(f"Error in processing loop: {e}")
            break

def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    sentiment = 1 if scores['pos'] > 0 else 0
    return sentiment


# Main script execution
def main():
    reddit_url = "https://www.reddit.com/r/bindingofisaac.json"
    items_url = 'https://tboi.com/all-items.json'
    headers = {"User-Agent": "Mozilla/5.0"}
    
    # Fetch and process Reddit data
    reddit_data = fetch_reddit_data(reddit_url, headers)
    if reddit_data:
        process_reddit_comments(reddit_data, headers)
    
    # Fetch and process items data
    items_html = fetch_items_data(items_url)
    if items_html:
        items = extract_item_titles(items_html)
        print(f"Extracted {len(items)} items.")

    # Preprocess the text in DataFrame
    df['reviewText'] = df['reviewText'].apply(preprocess_text)
    print(df)
    df['sentiment'] = df['reviewText'].apply(get_sentiment)
    print(df)

    print(confusion_matrix(df['Positive'], df['sentiment']))

    print(classification_report(df['Positive'], df['sentiment']))
    

if __name__ == "__main__":
    main()
