from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import warnings
import os
import logging
import traceback
import scipy.sparse
from book_metadata import (
    load_book_metadata, 
    create_content_similarity_matrix,
    load_or_create_cache,
    get_content_recommendations
)
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables to store data
books = None
moviemat = None
ratings = None
q_books = None
cosine_sim = None
tfidf_matrix = None
indices = None
books_metadata = None

def initialize_data():
    """Initialize all required data and handle any errors"""
    global books, moviemat, ratings, q_books, cosine_sim, tfidf_matrix, indices, books_metadata
    
    try:
        logger.info("Starting data initialization...")
        
        # Check if data file exists
        csv_path = "E:/final project/Graduation project/recommendation/Preprocessed_data.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found at: {csv_path}")
        
        # Load basic data first
        books, moviemat, ratings, q_books = load_data()
        
        if books is None or moviemat is None or ratings is None or q_books is None:
            raise ValueError("Failed to load basic data")
            
        logger.info("Basic data loaded successfully")
        logger.info(f"Books shape: {books.shape if books is not None else 'None'}")
        logger.info(f"Moviemat shape: {moviemat.shape if moviemat is not None else 'None'}")
        logger.info(f"Ratings shape: {ratings.shape if ratings is not None else 'None'}")
        logger.info(f"Q_books shape: {q_books.shape if q_books is not None else 'None'}")
        
        # Try to load content-based components
        try:
            cosine_sim, tfidf_matrix, indices, books_metadata = load_or_create_cache()
            if any(x is None for x in [cosine_sim, tfidf_matrix, indices, books_metadata]):
                logger.warning("Some content-based components are missing, attempting to recreate...")
                books_metadata = load_book_metadata()
                if books_metadata is not None:
                    cosine_sim, tfidf_matrix, indices = create_content_similarity_matrix(books_metadata)
        except Exception as e:
            logger.error(f"Error loading content-based components: {str(e)}")
            logger.error(traceback.format_exc())
            # Continue without content-based recommendations
            cosine_sim = None
            tfidf_matrix = None
            indices = None
            books_metadata = None
        
        logger.info("Data initialization completed")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize data: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def load_data():
    """Load and initialize basic data"""
    try:
        logger.info("Loading basic data...")
        
        # Read the CSV file
        csv_path = "E:/final project/Graduation project/recommendation/Preprocessed_data.csv"
        df = pd.read_csv(csv_path, 
                        sep=",", 
                        on_bad_lines='skip',
                        encoding='latin-1')
        
        # Verify required columns
        required_columns = ['user_id', 'book_title', 'rating']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Process data
        books = df[required_columns].iloc[:100000].copy()
        moviemat = books.pivot_table(index='user_id', columns='book_title', values='rating')
        ratings = pd.DataFrame(books.groupby('book_title')['rating'].mean())
        ratings['num of ratings'] = pd.DataFrame(books.groupby('book_title')['rating'].count())
        
        # Calculate weighted ratings
        m = ratings['num of ratings'].quantile(0.95)
        c = ratings['rating'].mean()
        
        def weight_rating(x, m=m, c=c):
            v = x['num of ratings']
            r = x['rating']
            return (v/(v+m)*r) + (m/(m+v)*c)
        
        q_books = ratings[ratings['num of ratings'] >= m].copy()
        q_books['score'] = q_books.apply(weight_rating, axis=1)
        
        return books, moviemat, ratings, q_books
        
    except Exception as e:
        logger.error(f"Error in load_data: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None, None

# Initialize data when starting the app
if not initialize_data():
    logger.error("Failed to initialize application data. Some features may not work correctly.")

def get_collaborative_recommendations(book_title, moviemat, ratings, min_ratings=100, n=5):
    """Get recommendations using collaborative filtering"""
    try:
        logger.info(f"Getting collaborative recommendations for: {book_title}")
        
        if book_title not in moviemat.columns:
            logger.warning(f"Book '{book_title}' not found in the dataset")
            return []
        
        # Get user ratings for the selected book
        book_user_ratings = moviemat[book_title]
        
        # Calculate correlation with other books
        similar_books = moviemat.corrwith(book_user_ratings)
        
        # Create correlation dataframe
        corr_book = pd.DataFrame(similar_books, columns=['Correlation'])
        corr_book.dropna(inplace=True)
        
        # Join with number of ratings
        corr_book = corr_book.join(ratings['num of ratings'])
        
        # Filter books with minimum ratings and sort by correlation
        recommendations = corr_book[corr_book['num of ratings'] > min_ratings].sort_values('Correlation', ascending=False)
        
        # Remove the input book from recommendations
        recommendations = recommendations[recommendations.index != book_title]
        
        # Get top n recommendations
        top_recommendations = recommendations.head(n)
        
        # Format recommendations with metadata
        recommendation_details = []
        for book, row in top_recommendations.iterrows():
            recommendation_details.append({
                'title': book,
                'correlation_score': float(row['Correlation']),
                'num_ratings': int(row['num of ratings']),
                'recommendation_type': 'Collaborative',
                'explanation': f"Recommended because users who liked '{book_title}' also rated this book highly"
            })
        
        logger.info(f"Found {len(recommendation_details)} collaborative recommendations")
        return recommendation_details
    
    except Exception as e:
        logger.error(f"Error getting collaborative recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def get_content_recommendations(book_title, cosine_sim, indices, books_metadata, n=5):
    """Get recommendations using content-based filtering"""
    try:
        logger.info(f"Getting content-based recommendations for: {book_title}")
        
        # Check each component
        if cosine_sim is None:
            logger.error("cosine_sim is None")
            return []
        if indices is None:
            logger.error("indices is None")
            return []
        if books_metadata is None:
            logger.error("books_metadata is None")
            return []
            
        # Log the state of our data
        logger.info(f"Data shapes - cosine_sim: {cosine_sim.shape if hasattr(cosine_sim, 'shape') else 'no shape'}, "
                   f"indices length: {len(indices) if indices is not None else 0}, "
                   f"books_metadata length: {len(books_metadata) if books_metadata is not None else 0}")
        
        # Check if book exists in indices
        if book_title not in indices:
            logger.warning(f"Book '{book_title}' not found in indices. Available books: {list(indices.keys())[:5]}...")
            return []
        
        # Get the index of the book
        idx = indices[book_title]
        logger.info(f"Found book at index {idx}")
        
        # Get similarity scores using sparse matrix
        if scipy.sparse.issparse(cosine_sim):
            # For sparse matrix, get the row and convert to dense array
            sim_scores = cosine_sim[idx].toarray().flatten()
            # Get indices of top n+1 scores (including the book itself)
            top_indices = sim_scores.argsort()[-n-1:][::-1]
            # Remove the book itself from recommendations
            top_indices = top_indices[top_indices != idx][:n]
            # Get the scores
            scores = sim_scores[top_indices]
            logger.info(f"Found {len(scores)} recommendations using sparse matrix")
        else:
            # For dense matrix, use the original approach
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:n+1]
            top_indices = [i for i, _ in sim_scores]
            scores = [score for _, score in sim_scores]
            logger.info(f"Found {len(scores)} recommendations using dense matrix")
        
        # Get metadata for recommendations
        recommendation_details = []
        for i, (book_idx, score) in enumerate(zip(top_indices, scores)):
            try:
                book_info = books_metadata.iloc[book_idx]
                recommendation_details.append({
                    'title': book_info['book_title'],
                    'similarity_score': float(score),
                    'recommendation_type': 'Content-Based',
                    'explanation': f"Recommended because it's similar to '{book_title}'"
                })
            except Exception as e:
                logger.error(f"Error processing recommendation {i}: {str(e)}")
                continue
        
        logger.info(f"Successfully created {len(recommendation_details)} content-based recommendations")
        return recommendation_details
    
    except Exception as e:
        logger.error(f"Error getting content-based recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def get_hybrid_recommendations(book_title, moviemat, ratings, cosine_sim, indices, books_metadata, alpha=0.5, min_ratings=100, n=5):
    """Get hybrid recommendations combining both approaches"""
    try:
        logger.info(f"Getting hybrid recommendations for: {book_title}")
        
        # Get collaborative filtering scores
        collab_recs = get_collaborative_recommendations(book_title, moviemat, ratings, min_ratings, n)
        
        # Get content-based recommendations
        content_recs = get_content_recommendations(book_title, cosine_sim, indices, books_metadata, n)
        
        if not collab_recs and not content_recs:
            logger.warning("No recommendations available from either method")
            return []
        
        if not collab_recs:
            logger.info("No collaborative recommendations, falling back to content-based")
            return content_recs
        
        if not content_recs:
            logger.info("No content-based recommendations, falling back to collaborative")
            return collab_recs
        
        # Combine recommendations
        combined_recs = {}
        
        # Add collaborative recommendations
        for rec in collab_recs:
            combined_recs[rec['title']] = {
                'title': rec['title'],
                'collab_score': rec['correlation_score'],
                'content_score': 0,
                'metadata': rec
            }
        
        # Add content-based recommendations
        for rec in content_recs:
            if rec['title'] in combined_recs:
                combined_recs[rec['title']]['content_score'] = rec['similarity_score']
                combined_recs[rec['title']]['metadata'].update(rec)
            else:
                combined_recs[rec['title']] = {
                    'title': rec['title'],
                    'collab_score': 0,
                    'content_score': rec['similarity_score'],
                    'metadata': rec
                }
        
        # Calculate hybrid scores
        for rec in combined_recs.values():
            # Normalize scores to [0, 1] range
            collab_score = (rec['collab_score'] + 1) / 2  # Convert from [-1, 1] to [0, 1]
            content_score = rec['content_score']  # Already in [0, 1] range
            
            rec['hybrid_score'] = 0.5 * collab_score + 0.5 * content_score
            rec['metadata']['recommendation_type'] = 'Hybrid'
            rec['metadata']['explanation'] = (
                f"Recommended based on both user preferences and content similarity to '{book_title}'"
            )
        
        # Sort by hybrid score and get top n
        final_recommendations = sorted(
            combined_recs.values(),
            key=lambda x: x['hybrid_score'],
            reverse=True
        )[:5]
        
        return jsonify({'recommendations': [rec['metadata'] for rec in final_recommendations]})
        
    except Exception as e:
        logger.error(f"Error in hybrid recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/')
def home():
    try:
        if ratings is None or q_books is None:
            return render_template('error.html', error="Data not properly loaded"), 500
            
        popular_books = ratings[ratings['num of ratings'] > 100].index.tolist()
        top_rated = q_books.sort_values('score', ascending=False).head(10).index.tolist()
        return render_template('index.html', books=popular_books, top_rated=top_rated)
    except Exception as e:
        logger.error(f"Error in home route: {str(e)}")
        logger.error(traceback.format_exc())
        return render_template('error.html', error="Failed to load home page"), 500

@app.route('/recommend/collaborative', methods=['POST'])
def recommend_collaborative():
    try:
        if moviemat is None or ratings is None:
            return jsonify({'error': 'Recommendation system not properly initialized'}), 500
            
        book_title = request.form.get('book_title')
        if not book_title:
            return jsonify({'error': 'No book selected'}), 400
            
        if book_title not in moviemat.columns:
            return jsonify({'error': f'Book "{book_title}" not found in database'}), 404
        
        recommendations = get_collaborative_recommendations(book_title, moviemat, ratings)
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        logger.error(f"Error in collaborative recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/recommend/content', methods=['POST'])
def recommend_content():
    try:
        if any(x is None for x in [cosine_sim, indices, books_metadata]):
            return jsonify({'error': 'Content-based recommendations not available'}), 503
            
        book_title = request.form.get('book_title')
        if not book_title:
            return jsonify({'error': 'No book selected'}), 400
            
        if book_title not in indices:
            return jsonify({'error': f'Book "{book_title}" not found in database'}), 404
        
        recommendations = get_content_recommendations(book_title, cosine_sim, indices, books_metadata)
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        logger.error(f"Error in content-based recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/recommend/hybrid', methods=['POST'])
def recommend_hybrid():
    try:
        if moviemat is None or ratings is None:
            return jsonify({'error': 'Collaborative filtering not available'}), 503
        if any(x is None for x in [cosine_sim, indices, books_metadata]):
            return jsonify({'error': 'Content-based filtering not available'}), 503
            
        book_title = request.form.get('book_title')
        if not book_title:
            return jsonify({'error': 'No book selected'}), 400
            
        if book_title not in moviemat.columns or book_title not in indices:
            return jsonify({'error': f'Book "{book_title}" not found in database'}), 404
        
        collab_recs = get_collaborative_recommendations(book_title, moviemat, ratings)
        content_recs = get_content_recommendations(book_title, cosine_sim, indices, books_metadata)
        
        if not collab_recs and not content_recs:
            return jsonify({'error': 'No recommendations available'}), 404
        
        if not collab_recs:
            return jsonify({'recommendations': content_recs})
        
        if not content_recs:
            return jsonify({'recommendations': collab_recs})
        
        combined_recs = {}
        
        for rec in collab_recs:
            combined_recs[rec['title']] = {
                'title': rec['title'],
                'collab_score': rec['correlation_score'],
                'content_score': 0,
                'metadata': rec
            }
        
        for rec in content_recs:
            if rec['title'] in combined_recs:
                combined_recs[rec['title']]['content_score'] = rec['similarity_score']
                combined_recs[rec['title']]['metadata'].update(rec)
            else:
                combined_recs[rec['title']] = {
                    'title': rec['title'],
                    'collab_score': 0,
                    'content_score': rec['similarity_score'],
                    'metadata': rec
                }
        
        for rec in combined_recs.values():
            collab_score = (rec['collab_score'] + 1) / 2
            content_score = rec['content_score']
            rec['hybrid_score'] = 0.5 * collab_score + 0.5 * content_score
            rec['metadata']['recommendation_type'] = 'Hybrid'
            rec['metadata']['explanation'] = (
                f"Recommended based on both user preferences and content similarity to '{book_title}'"
            )
        
        final_recommendations = sorted(
            combined_recs.values(),
            key=lambda x: x['hybrid_score'],
            reverse=True
        )[:5]
        
        return jsonify({'recommendations': [rec['metadata'] for rec in final_recommendations]})
        
    except Exception as e:
        logger.error(f"Error in hybrid recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5003) 