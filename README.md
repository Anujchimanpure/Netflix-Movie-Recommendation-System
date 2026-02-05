# Netflix-Movie-Recommendation-System

A content-based movie recommendation system inspired by Netflix, built using Machine Learning, Python, and Streamlit.
The system recommends movies similar to a user-entered title based on textual similarity and enriches results with live poster data.


With the rapid growth of digital streaming platforms, users are often overwhelmed by the vast amount of available content.
This project aims to recommend relevant movies to users by analyzing similarities between movie descriptions and metadata.

The system:
-Accepts a movie name as input
-Identifies similar movies using a similarity model
-Displays recommendations in a Netflix-style UI
-Fetches movie posters dynamically using the OMDb API

# Features

-Content-based movie recommendation

-Case-insensitive, user-friendly search

-Live movie posters via OMDb API

-Cosine similarity–based recommendation engine

-Netflix-inspired UI using Streamlit + CSS

-Graceful handling of missing posters


# Technologies Used

Python
Pandas & NumPy – data processing
Scikit-learn – similarity computation
Streamlit – web application UI

# Dataset Details

Based on the Netflix Movies & TV Shows dataset
Cleaned and extended manually for demonstration

Includes:
  Title
  Description
  Cast
  Director
  Genre
  Duration
  Additional derived features:
  duration_value
  duration_type


# Note !!

“The dataset was intentionally kept limited to focus on recommendation logic and similarity modeling.
The system is designed to scale seamlessly with larger or real-time datasets.”

OMDb API – poster and metadata fetching
