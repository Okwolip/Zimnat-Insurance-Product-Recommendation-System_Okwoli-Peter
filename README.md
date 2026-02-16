# Zimnat Insurance Product Recommendation System

# Project Overview

This project implements a machine learning-based recommendation system designed to predict the 'Top 5 most suitable insurance products' for customers based on demographic, behavioral, and product interaction features.

The system was developed as part of a capstone project for DeepTech/ DSN data science & Machine Learning project, and it demonstrates end-to-end:

. Data preprocessing
. Feature engineering
. Multi-label classification
. Model calibration
. Deployment using Streamlit Cloud

#  Live Application

 Deployed App:  
https://zimnat-insurance-appuct-recommendation-systemokwoli-peter-kpqx.streamlit.app/

# Problem Statement

Insurance institutions often struggle with:

. Low cross-selling performance
. Poor product targeting
. Limited personalization

This system addresses the challenges by leveraging:

. Branch-level behavioral metrics
. Occupation-based segmentation
. Product co-occurrence relationships

to generate intelligent product recommendations.


# Model Architecture

The system uses:

. One-vs-Rest multi-label classification strategy
. LightGBM models per product
. Feature frequency encoding
. Co-occurrence-based relationship metrics
. Probability ranking for top-5 recommendation selection

# Key Features Used

. Age
. Days Since Joining
. Number of Existing Products
. Branch Popularity Score
. Occupation Frequency Score
. Occupation Category Score
. Product Co-occurrence Sum
. Product Co-occurrence Max

#  Deployment

The application is deployed using:

. Streamlit
. GitHub
. Streamlit Community Cloud
