"""
Regenerate corrupted model files
"""
import pandas as pd
import joblib
import os
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

print("Starting model regeneration...")
print("=" * 50)

# Step 1: Load and prepare movies data
print("\n1. Loading movies data...")
movies_candidates = ['movie-project/cleaned_movies.csv', 'cleaned_movies.csv', 'archive/tmdb_5000_movies.csv']
movies_path = next((p for p in movies_candidates if os.path.exists(p)), None)

if movies_path:
    print(f"   ✓ Found movies data: {movies_path}")
    movies = pd.read_csv(movies_path)
    movies_title = movies[['id', 'title', 'genres', 'cast', 'crew', 'overview', 'vote_average', 'release_date']].head(5000)
    joblib.dump(movies_title, 'Model/movies_data.joblib')
    print(f"   ✓ Saved movies_data.joblib ({len(movies_title)} movies)")
else:
    print("   ✗ Movies data not found")

# Step 2: Create similarity matrix (using TMDB data)
print("\n2. Creating similarity matrix...")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if movies_path:
    # Create TF-IDF matrix for overviews
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    
    # Fill NaN values in overview with empty string
    overviews = movies_title['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(overviews)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix)
    joblib.dump(similarity, 'Model/similarity.joblib')
    print(f"   ✓ Saved similarity.joblib (shape: {similarity.shape})")
else:
    print("   ✗ Cannot create similarity matrix without movies data")

# Step 3: Train sentiment analysis model
print("\n3. Training sentiment analysis model...")
imdb_candidates = ['movie-project/IMDB Dataset.csv', 'IMDB Dataset.csv']
imdb_path = next((p for p in imdb_candidates if os.path.exists(p)), None)

if imdb_path is None:
    print('   ⚠️  IMDB Dataset.csv not found.')
    print('   Download from: https://www.kaggle.com/datasets/marcoiarpa/imdb-review-dataset')
else:
    print(f'   ✓ Found IMDB dataset: {imdb_path}')
    print('   Loading and preprocessing (this may take a minute)...')
    
    imdb_df = pd.read_csv(imdb_path)
    imdb_df['label'] = (imdb_df['sentiment'] == 'positive').astype(int)
    
    def preprocess(text):
        t = str(text).lower()
        t = re.sub(r'<.*?>', '', t)
        t = t.translate(str.maketrans('', '', string.punctuation))
        t = re.sub(r'\d+', '', t)
        return re.sub(r'\s+', ' ', t).strip()
    
    imdb_df['clean_review'] = imdb_df['review'].apply(preprocess)
    
    print('   Creating TF-IDF vectorizer...')
    tfidf_sentiment = TfidfVectorizer(max_features=10000, stop_words='english')
    X = tfidf_sentiment.fit_transform(imdb_df['clean_review'])
    y = imdb_df['label']
    
    print('   Splitting data...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print('   Training logistic regression...')
    lr = LogisticRegression(max_iter=200, n_jobs=-1)
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)
    
    print('\n   === Logistic Regression Results ===')
    print(f'   Accuracy: {accuracy_score(y_test, preds):.4f}')
    print('\n   Classification Report:')
    print(classification_report(y_test, preds, target_names=['Negative', 'Positive']))
    
    joblib.dump(lr, 'Model/sentiment_analysis_model.pkl')
    joblib.dump(tfidf_sentiment, 'Model/tfidf_vectorizer.pkl')
    print('\n   ✓ Sentiment models saved successfully')

print("\n" + "=" * 50)
print("✓ Model regeneration complete!")
print("=" * 50)
