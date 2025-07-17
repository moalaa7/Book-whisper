import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import logging
import traceback
import os
import pickle
from scipy import sparse
import scipy

# Configure logging
logger = logging.getLogger(__name__)

# Constants for caching
CACHE_DIR = "cache"
SIMILARITY_CACHE_FILE = os.path.join(CACHE_DIR, "similarity_matrix.pkl")
TFIDF_CACHE_FILE = os.path.join(CACHE_DIR, "tfidf_matrix.pkl")
INDICES_CACHE_FILE = os.path.join(CACHE_DIR, "indices.pkl")
METADATA_CACHE_FILE = os.path.join(CACHE_DIR, "metadata.pkl")

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

def load_or_create_cache():
    """Load cached data or return None if not available"""
    try:
        cache_dir = "cache"
        if not os.path.exists(cache_dir):
            logger.info("Cache directory does not exist")
            return None, None, None, None
            
        required_files = ['cosine_sim.pkl', 'tfidf_matrix.pkl', 'indices.pkl', 'books_metadata.pkl']
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(cache_dir, f))]
        
        if missing_files:
            logger.info(f"Missing cache files: {missing_files}")
            return None, None, None, None
            
        logger.info("Loading cache files...")
        try:
            with open(os.path.join(cache_dir, 'cosine_sim.pkl'), 'rb') as f:
                cosine_sim = pickle.load(f)
            with open(os.path.join(cache_dir, 'tfidf_matrix.pkl'), 'rb') as f:
                tfidf_matrix = pickle.load(f)
            with open(os.path.join(cache_dir, 'indices.pkl'), 'rb') as f:
                indices = pickle.load(f)
            with open(os.path.join(cache_dir, 'books_metadata.pkl'), 'rb') as f:
                books_metadata = pickle.load(f)
                
            logger.info("Successfully loaded all cache files")
            return cosine_sim, tfidf_matrix, indices, books_metadata
            
        except Exception as e:
            logger.error(f"Error loading cache files: {str(e)}")
            logger.error(traceback.format_exc())
            return None, None, None, None
            
    except Exception as e:
        logger.error(f"Error in load_or_create_cache: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None, None

def save_to_cache(cosine_sim, tfidf_matrix, indices, books_metadata):
    """Save computed matrices to cache"""
    try:
        logger.info("Saving computed matrices to cache...")
        with open(SIMILARITY_CACHE_FILE, 'wb') as f:
            pickle.dump(cosine_sim, f)
        with open(TFIDF_CACHE_FILE, 'wb') as f:
            pickle.dump(tfidf_matrix, f)
        with open(INDICES_CACHE_FILE, 'wb') as f:
            pickle.dump(indices, f)
        with open(METADATA_CACHE_FILE, 'wb') as f:
            pickle.dump(books_metadata, f)
        logger.info("Cache saved successfully")
    except Exception as e:
        logger.error(f"Error saving cache: {str(e)}")

def load_book_metadata():
    """Load and process book metadata"""
    try:
        logger.info("Starting to load book metadata...")
        
        # Try to load from cache first
        logger.info("Attempting to load from cache...")
        cosine_sim, tfidf_matrix, indices, books_metadata = load_or_create_cache()
        if cosine_sim is not None and tfidf_matrix is not None and indices is not None and books_metadata is not None:
            logger.info("Successfully loaded metadata from cache")
            logger.info(f"Cache data shapes - cosine_sim: {cosine_sim.shape}, tfidf_matrix: {tfidf_matrix.shape}, indices: {len(indices)}, books_metadata: {len(books_metadata)}")
            return books_metadata
        else:
            logger.info("Cache load failed or incomplete, will create new data")
        
        # Load the main dataset with only necessary columns
        logger.info("Reading CSV file for metadata...")
        csv_path = "E:/final project/Graduation project/recommendation/Preprocessed_data.csv"
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found at path: {csv_path}")
            return None
            
        try:
            # Read only necessary columns to reduce memory usage
            df = pd.read_csv(csv_path, 
                            sep=",", 
                            usecols=['book_title', 'user_id', 'rating'],  # Only essential columns
                            on_bad_lines='skip',
                            encoding='latin-1')
            
            logger.info(f"CSV loaded successfully. Shape: {df.shape}")
            logger.info(f"Columns in dataframe: {df.columns.tolist()}")
            
            # Check for null values
            null_counts = df.isnull().sum()
            logger.info(f"Null value counts:\n{null_counts}")
            
            # Check for empty strings in book titles
            empty_titles = df['book_title'].str.strip().eq('').sum()
            if empty_titles > 0:
                logger.warning(f"Found {empty_titles} empty book titles")
                df = df[df['book_title'].str.strip().ne('')]
            
        except Exception as e:
            logger.error(f"Error reading CSV file: {str(e)}")
            logger.error(traceback.format_exc())
            return None
        
        # Create a unique books dataframe with only popular books (more than 10 ratings)
        logger.info("Filtering popular books...")
        try:
            book_ratings = df.groupby('book_title').size()
            logger.info(f"Total unique books before filtering: {len(book_ratings)}")
            
            # Get books with at least 10 ratings
            popular_books = book_ratings[book_ratings >= 10].index.tolist()
            logger.info(f"Number of popular books (>= 10 ratings): {len(popular_books)}")
            
            if len(popular_books) == 0:
                logger.error("No books found with 10 or more ratings")
                return None
                
            books_metadata = pd.DataFrame({'book_title': popular_books})
            logger.info(f"Created metadata dataframe with {len(books_metadata)} books")
            
            # Verify book titles are unique
            if len(books_metadata['book_title'].unique()) != len(books_metadata):
                logger.warning("Duplicate book titles found, removing duplicates")
                books_metadata = books_metadata.drop_duplicates(subset=['book_title'])
                logger.info(f"Metadata dataframe after removing duplicates: {len(books_metadata)} books")
            
        except Exception as e:
            logger.error(f"Error filtering popular books: {str(e)}")
            logger.error(traceback.format_exc())
            return None
        
        # Create simple features using book titles
        logger.info("Creating features from book titles...")
        try:
            # Clean and normalize book titles
            books_metadata['features'] = (books_metadata['book_title']
                                        .str.lower()
                                        .str.strip()
                                        .str.replace(r'[^\w\s]', ' ', regex=True)  # Replace special chars with space
                                        .str.replace(r'\s+', ' ', regex=True))  # Replace multiple spaces with single space
            
            # Remove any empty features
            empty_features = books_metadata['features'].str.strip().eq('').sum()
            if empty_features > 0:
                logger.warning(f"Found {empty_features} empty features, removing these books")
                books_metadata = books_metadata[books_metadata['features'].str.strip().ne('')]
            
            logger.info(f"Sample features:\n{books_metadata['features'].head(3)}")
            
        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            logger.error(traceback.format_exc())
            return None
        
        # Create and cache the similarity matrix
        logger.info("Creating similarity matrix...")
        try:
            cosine_sim, tfidf_matrix, indices = create_content_similarity_matrix(books_metadata)
            
            if cosine_sim is None or tfidf_matrix is None or indices is None:
                logger.error("Failed to create similarity matrix - one or more components are None")
                return None
                
            logger.info(f"Successfully created similarity components:")
            logger.info(f"- cosine_sim shape: {cosine_sim.shape}")
            logger.info(f"- tfidf_matrix shape: {tfidf_matrix.shape}")
            logger.info(f"- indices length: {len(indices)}")
            
            # Verify indices match books_metadata
            if len(indices) != len(books_metadata):
                logger.error(f"Indices length ({len(indices)}) doesn't match books_metadata length ({len(books_metadata)})")
                return None
            
        except Exception as e:
            logger.error(f"Error creating similarity matrix: {str(e)}")
            logger.error(traceback.format_exc())
            return None
        
        # Save to cache
        try:
            logger.info("Attempting to save to cache...")
            save_to_cache(cosine_sim, tfidf_matrix, indices, books_metadata)
            logger.info("Successfully saved to cache")
        except Exception as e:
            logger.error(f"Failed to save to cache: {str(e)}")
            logger.error(traceback.format_exc())
            # Continue even if cache save fails
        
        logger.info("Book metadata loaded successfully")
        return books_metadata
        
    except Exception as e:
        logger.error(f"Error in load_book_metadata: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def create_content_similarity_matrix(books_metadata):
    """Create similarity matrix for content-based recommendations"""
    try:
        logger.info("Creating content similarity matrix...")
        
        if books_metadata is None or len(books_metadata) == 0:
            logger.error("No book metadata available")
            return None, None, None
            
        logger.info(f"Input books_metadata shape: {books_metadata.shape}")
        logger.info(f"Sample book titles:\n{books_metadata['book_title'].head(3)}")
        
        # Verify features column exists
        if 'features' not in books_metadata.columns:
            logger.error("'features' column not found in books_metadata")
            return None, None, None
            
        # Check for empty features
        empty_features = books_metadata['features'].isnull().sum()
        if empty_features > 0:
            logger.error(f"Found {empty_features} empty features")
            return None, None, None
            
        # Create TF-IDF matrix with minimal features
        logger.info("Creating TF-IDF matrix...")
        tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=500,  # Reduced features
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),
            strip_accents='unicode',
            lowercase=True
        )
        
        try:
            # Use only book titles for features
            features = books_metadata['features'].astype(str)
            logger.info(f"Sample features before TF-IDF:\n{features.head(3)}")
            
            tfidf_matrix = tfidf.fit_transform(features)
            logger.info(f"TF-IDF matrix created. Shape: {tfidf_matrix.shape}")
            
            # Log some feature names
            feature_names = tfidf.get_feature_names_out()
            logger.info(f"Number of features: {len(feature_names)}")
            logger.info(f"Sample features: {feature_names[:10]}")
            
            # Check if matrix is empty
            if tfidf_matrix.shape[0] == 0 or tfidf_matrix.shape[1] == 0:
                logger.error("TF-IDF matrix is empty")
                return None, None, None
                
        except Exception as e:
            logger.error(f"Error creating TF-IDF matrix: {str(e)}")
            logger.error(traceback.format_exc())
            return None, None, None
        
        # Calculate cosine similarity using sparse matrix
        logger.info("Calculating cosine similarity...")
        try:
            # Convert to sparse matrix for faster computation
            tfidf_sparse = tfidf_matrix.tocsr()
            logger.info("Converted to sparse matrix")
            
            # Calculate similarity
            cosine_sim = cosine_similarity(tfidf_sparse, dense_output=False)
            logger.info(f"Similarity matrix created. Shape: {cosine_sim.shape}")
            
            # Verify similarity matrix
            if cosine_sim.shape[0] != cosine_sim.shape[1]:
                logger.error(f"Similarity matrix is not square: {cosine_sim.shape}")
                return None, None, None
                
            if cosine_sim.shape[0] != len(books_metadata):
                logger.error(f"Similarity matrix size ({cosine_sim.shape[0]}) doesn't match number of books ({len(books_metadata)})")
                return None, None, None
                
            # Check for NaN values
            if scipy.sparse.issparse(cosine_sim):
                if np.isnan(cosine_sim.data).any():
                    logger.error("Found NaN values in similarity matrix")
                    return None, None, None
            else:
                if np.isnan(cosine_sim).any():
                    logger.error("Found NaN values in similarity matrix")
                    return None, None, None
                    
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            logger.error(traceback.format_exc())
            return None, None, None
        
        # Create indices mapping
        logger.info("Creating book title indices...")
        try:
            indices = pd.Series(books_metadata.index, index=books_metadata['book_title']).drop_duplicates()
            logger.info(f"Created indices for {len(indices)} books")
            
            # Verify indices
            if len(indices) != len(books_metadata):
                logger.error(f"Indices length ({len(indices)}) doesn't match books_metadata length ({len(books_metadata)})")
                return None, None, None
                
            # Check for duplicate book titles
            duplicate_titles = books_metadata['book_title'].duplicated().sum()
            if duplicate_titles > 0:
                logger.error(f"Found {duplicate_titles} duplicate book titles")
                return None, None, None
                
            # Log some sample indices
            sample_titles = list(indices.index[:3])
            logger.info(f"Sample book titles in indices: {sample_titles}")
            
        except Exception as e:
            logger.error(f"Error creating indices: {str(e)}")
            logger.error(traceback.format_exc())
            return None, None, None
        
        logger.info("Content similarity matrix created successfully")
        return cosine_sim, tfidf_matrix, indices
        
    except Exception as e:
        logger.error(f"Error in create_content_similarity_matrix: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None

def get_content_recommendations(book_title, cosine_sim, indices, books_metadata, n=5):
    """Get recommendations using content-based filtering"""
    try:
        logger.info(f"Getting content-based recommendations for: {book_title}")
        
        if cosine_sim is None or indices is None or books_metadata is None:
            logger.warning("Content-based recommendations not available")
            return []
            
        if book_title not in indices:
            logger.warning(f"Book '{book_title}' not found in indices")
            return []
        
        # Get the index of the book
        idx = indices[book_title]
        
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
        else:
            # For dense matrix, use the original approach
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:n+1]
            top_indices = [i for i, _ in sim_scores]
            scores = [score for _, score in sim_scores]
        
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
        
        logger.info(f"Found {len(recommendation_details)} content-based recommendations")
        return recommendation_details
    
    except Exception as e:
        logger.error(f"Error getting content-based recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def get_hybrid_recommendations(book_title, collab_scores, content_scores, alpha=0.5):
    """
    Combine collaborative and content-based recommendations
    alpha: weight for collaborative filtering (1-alpha for content-based)
    """
    try:
        logger.info(f"Getting hybrid recommendations for: {book_title}")
        
        # Normalize scores
        collab_scores = (collab_scores - collab_scores.min()) / (collab_scores.max() - collab_scores.min())
        content_scores = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min())
        
        # Combine scores
        hybrid_scores = alpha * collab_scores + (1 - alpha) * content_scores
        
        # Sort and get top recommendations
        recommendations = pd.Series(hybrid_scores, index=collab_scores.index)
        recommendations = recommendations.sort_values(ascending=False)
        
        # Remove the input book
        recommendations = recommendations[recommendations.index != book_title]
        
        logger.info(f"Found {len(recommendations.head(5))} hybrid recommendations")
        return recommendations.head(5).index.tolist()
        
    except Exception as e:
        logger.error(f"Error getting hybrid recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        return [] 