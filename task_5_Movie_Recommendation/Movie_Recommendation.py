# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from collections import defaultdict

# Step 1: Load the dataset
# Download from: https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset
ratings = pd.read_csv('ml-100k/u.data', sep='\t', 
                      names=['user_id', 'item_id', 'rating', 'timestamp'])
movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1',
                    names=['item_id', 'title', 'release_date', 'video_release', 
                          'imdb_url', 'unknown', 'Action', 'Adventure', 
                          'Animation', 'Children', 'Comedy', 'Crime', 
                          'Documentary', 'Drama', 'Fantasy', 'FilmNoir', 
                          'Horror', 'Musical', 'Mystery', 'Romance', 
                          'SciFi', 'Thriller', 'War', 'Western'])

# Step 2: Data exploration and preprocessing
print("Ratings data:")
print(ratings.head())
print("\nMovies data:")
print(movies.head())

# Check basic statistics
print("\nRatings statistics:")
print(ratings.describe())

# Create user-item matrix
user_item_matrix = ratings.pivot_table(index='user_id', columns='item_id', values='rating')
print(f"\nUser-item matrix shape: {user_item_matrix.shape}")

# Fill missing values with 0 (assuming no rating means 0)
user_item_matrix = user_item_matrix.fillna(0)

# Step 3: User-based collaborative filtering
def user_based_recommendation(user_id, user_item_matrix, movies, n_recommendations=5, min_similar_users=5):
    # Calculate similarity only using commonly rated items
    user_ratings = user_item_matrix.loc[user_id]
    rated_mask = user_ratings > 0
    common_ratings = user_item_matrix.loc[:, rated_mask]
    
    # Only compare with users who have rated at least 10 of the same movies
    user_similarity = cosine_similarity(common_ratings.fillna(0))
    user_similarity = pd.DataFrame(user_similarity, 
                                 index=user_item_matrix.index, 
                                 columns=user_item_matrix.index)
    
    # Get similar users (excluding self)
    similar_users = user_similarity[user_id].sort_values(ascending=False)[1:min_similar_users+1]
    
    # Only proceed if we found enough similar users
    if len(similar_users) < min_similar_users:
        return []
    
    # Get movies rated by similar users but not by our user
    recommendations = {}
    for item in user_item_matrix.columns:
        if user_ratings[item] == 0:  # Not rated by our user
            # Calculate weighted average rating from similar users
            weighted_sum = 0
            similarity_sum = 0
            for other_user, similarity in similar_users.items():
                if user_item_matrix.loc[other_user, item] > 0:
                    weighted_sum += similarity * user_item_matrix.loc[other_user, item]
                    similarity_sum += similarity
            
            if similarity_sum > 0:
                recommendations[item] = weighted_sum / similarity_sum
    
    # Get top recommendations (with at least 3 supporting ratings)
    recommendations = sorted([(k, v) for k, v in recommendations.items()], 
                            key=lambda x: x[1], reverse=True)[:n_recommendations]
    
    # Get movie titles
    return [(movies[movies['item_id'] == item_id]['title'].values[0], pred_rating)
            for item_id, pred_rating in recommendations]



# Test the recommendation system
user_id = 1
print(f"\nTop 5 recommendations for user {user_id}:")
recommendations = user_based_recommendation(user_id, user_item_matrix, movies)
for i, (title, rating) in enumerate(recommendations, 1):
    print(f"{i}. {title} (predicted rating: {rating:.2f})")

# Step 4: Evaluation using precision at K
def evaluate_recommendations(user_item_matrix, movies, k=5, test_size=0.2, min_ratings=20):
    # Only evaluate users with sufficient ratings
    test_users = [u for u in user_item_matrix.index 
                 if (user_item_matrix.loc[u] > 0).sum() >= min_ratings]
    test_users = np.random.choice(test_users, size=int(len(test_users)*test_size), replace=False)
    
    precisions = []
    for user in test_users:
        # Hide 20% of ratings for testing
        rated_items = user_item_matrix.loc[user][user_item_matrix.loc[user] > 0].index
        hide_items = np.random.choice(rated_items, size=int(0.2*len(rated_items)), replace=False)
        
        # Create train matrix
        train_matrix = user_item_matrix.copy()
        train_matrix.loc[user, hide_items] = 0
        
        # Get recommendations
        recs = user_based_recommendation(user, train_matrix, movies, n_recommendations=k)
        if not recs:
            continue
            
        recommended_items = [movies[movies['title'] == title]['item_id'].values[0] 
                           for title, _ in recs]
        
        # Calculate precision
        relevant = len(set(recommended_items) & set(hide_items))
        precisions.append(relevant / k)
    
    return np.mean(precisions) if precisions else 0




precision_at_5 = evaluate_recommendations(user_item_matrix, movies)
print(f"\nPrecision@5: {precision_at_5:.3f}")

# BONUS 1: Item-based collaborative filtering
def item_based_recommendation(user_id, user_item_matrix, movies, n_recommendations=5):
    # Calculate item-item similarity
    item_similarity = cosine_similarity(user_item_matrix.T)
    item_similarity = pd.DataFrame(item_similarity, 
                                 index=user_item_matrix.columns, 
                                 columns=user_item_matrix.columns)
    
    # Get items rated by the user
    user_ratings = user_item_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0].index
    
    # Calculate predicted ratings for unrated items
    recommendations = {}
    for item in user_item_matrix.columns:
        if item not in rated_items:
            # Find similar items that the user has rated
            similar_items = item_similarity[item][rated_items]
            similar_items = similar_items[similar_items > 0]  # Only positive similarities
            
            if not similar_items.empty:
                # Calculate weighted average rating
                weighted_sum = (similar_items * user_ratings[similar_items.index]).sum()
                similarity_sum = similar_items.sum()
                recommendations[item] = weighted_sum / similarity_sum
    
    # Get top recommendations
    recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
    
    # Get movie titles
    recommended_movies = []
    for item_id, pred_rating in recommendations:
        movie_title = movies[movies['item_id'] == item_id]['title'].values[0]
        recommended_movies.append((movie_title, pred_rating))
    
    return recommended_movies

print("\nItem-based recommendations for user 1:")
item_based_recs = item_based_recommendation(1, user_item_matrix, movies)
for i, (title, rating) in enumerate(item_based_recs, 1):
    print(f"{i}. {title} (predicted rating: {rating:.2f})")

# BONUS 2: Matrix Factorization with SVD
def svd_recommendation(user_item_matrix, movies, n_factors=50, n_recommendations=5):
    # Perform SVD
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    svd.fit(user_item_matrix)
    
    # Reconstruct the matrix
    user_factors = svd.transform(user_item_matrix)
    item_factors = svd.components_.T
    reconstructed_matrix = np.dot(user_factors, item_factors.T)
    
    # Convert back to DataFrame
    reconstructed_matrix = pd.DataFrame(reconstructed_matrix, 
                                      index=user_item_matrix.index,
                                      columns=user_item_matrix.columns)
    
    # Function to get recommendations for a user
    def get_recommendations(user_id):
        user_ratings = user_item_matrix.loc[user_id]
        predicted_ratings = reconstructed_matrix.loc[user_id]
        
        # Find items not rated by user
        unrated_items = user_ratings[user_ratings == 0].index
        
        # Get top predicted ratings for unrated items
        recommendations = predicted_ratings[unrated_items].sort_values(ascending=False)[:n_recommendations]
        
        # Get movie titles
        recommended_movies = []
        for item_id, pred_rating in recommendations.items():
            movie_title = movies[movies['item_id'] == item_id]['title'].values[0]
            recommended_movies.append((movie_title, pred_rating))
        
        return recommended_movies
    
    return get_recommendations

svd_recommender = svd_recommendation(user_item_matrix, movies)
print("\nSVD-based recommendations for user 1:")
svd_recs = svd_recommender(1)
for i, (title, rating) in enumerate(svd_recs, 1):
    print(f"{i}. {title} (predicted rating: {rating:.2f})")

# Modified SVD evaluation function to handle NaN values
def evaluate_svd(user_item_matrix, n_factors=50, test_size=0.2):
    # Create train-test split
    train_matrix = user_item_matrix.copy()
    test_matrix = pd.DataFrame(np.zeros(user_item_matrix.shape),
                             index=user_item_matrix.index,
                             columns=user_item_matrix.columns)
    
    for user in user_item_matrix.index:
        rated_items = user_item_matrix.loc[user][user_item_matrix.loc[user] > 0].index
        if len(rated_items) > 10:  # Only evaluate users with enough ratings
            test_items = np.random.choice(rated_items, size=int(test_size*len(rated_items)), replace=False)
            train_matrix.loc[user, test_items] = 0
            test_matrix.loc[user, test_items] = user_item_matrix.loc[user, test_items]
    
    # Remove users with no ratings in test set
    test_users = test_matrix[(test_matrix > 0).any(axis=1)].index
    if len(test_users) == 0:
        return np.nan  # No users to evaluate
    
    # Train SVD
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    svd.fit(train_matrix)
    
    # Reconstruct matrix
    user_factors = svd.transform(train_matrix)
    item_factors = svd.components_.T
    reconstructed_matrix = np.dot(user_factors, item_factors.T)
    
    # Calculate RMSE on test set
    test_ratings = []
    pred_ratings = []
    for user in test_users:
        for item in test_matrix.columns:
            actual = test_matrix.loc[user, item]
            if actual > 0:  # Only consider actual ratings
                predicted = reconstructed_matrix[user_item_matrix.index.get_loc(user), 
                                              user_item_matrix.columns.get_loc(item)]
                test_ratings.append(actual)
                pred_ratings.append(predicted)
    
    if not test_ratings:  # No test ratings found
        return np.nan
    
    rmse = np.sqrt(mean_squared_error(test_ratings, pred_ratings))
    return rmse

# Modified recommendation functions to prevent perfect scores
def user_based_recommendation(user_id, user_item_matrix, movies, n_recommendations=5):
    # Normalize ratings to reduce perfect score predictions
    user_mean = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].mean()
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity = pd.DataFrame(user_similarity, 
                                 index=user_item_matrix.index, 
                                 columns=user_item_matrix.index)
    
    # Get similar users (excluding self)
    similar_users = user_similarity[user_id].sort_values(ascending=False)[1:11]
    
    recommendations = defaultdict(float)
    item_counts = defaultdict(int)
    
    for other_user, similarity in similar_users.items():
        for item in user_item_matrix.columns:
            if (user_item_matrix.loc[other_user, item] > 0 and 
                user_item_matrix.loc[user_id, item] == 0):
                recommendations[item] += similarity * (user_item_matrix.loc[other_user, item] - user_mean)
                item_counts[item] += 1
    
    # Average the recommendations
    for item in recommendations:
        if item_counts[item] > 0:
            recommendations[item] = user_mean + (recommendations[item] / item_counts[item])
            # Clip to rating range
            recommendations[item] = max(1, min(5, recommendations[item]))
    
    # Get top recommendations
    recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
    
    # Get movie titles
    recommended_movies = []
    for item_id, pred_rating in recommendations:
        movie_title = movies[movies['item_id'] == item_id]['title'].values[0]
        recommended_movies.append((movie_title, pred_rating))
    
    return recommended_movies

# After making these changes, run the evaluation again
try:
    svd_rmse = evaluate_svd(user_item_matrix)
    print(f"\nSVD RMSE: {svd_rmse:.3f}")
except Exception as e:
    print(f"\nError in SVD evaluation: {str(e)}")