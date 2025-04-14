# Career Recommendation - Random Forest Model
# This notebook builds a Random Forest classifier to recommend careers based on psychometric assessment data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from collections import Counter
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('psychometric_dataset.csv')

# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Check class distribution
job_counts = Counter(df['Job_Role'])
print("\nJob role distribution:")
for job, count in job_counts.items():
    print(f"{job}: {count}")

# Find jobs with only one entry
single_entry_jobs = [job for job, count in job_counts.items() if count == 1]
print(f"\nJobs with only one entry: {single_entry_jobs}")

# For the model to work properly, we need to either:
# 1. Remove classes with only one member, or
# 2. Use simple train_test_split without stratification

# Option 1: Remove single-entry jobs
if single_entry_jobs:
    print(f"\nRemoving {len(single_entry_jobs)} job roles with only one entry")
    df = df[~df['Job_Role'].isin(single_entry_jobs)]
    print(f"New dataset shape: {df.shape}")

# Separate features and target variable
X = df.drop('Job_Role', axis=1)
y = df['Job_Role']

# Split the data into training and testing sets
# Now we can use stratification since all classes have at least 2 members
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Testing set shape: {X_test.shape}, {y_test.shape}")

# Create a pipeline with scaling and RandomForest
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=42))
])

# Define parameter grid for GridSearchCV
# Using a smaller parameter grid for faster execution
param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [None, 15],
    'rf__min_samples_split': [2, 5],
    'rf__max_features': ['sqrt', 'log2']
}

# Perform GridSearchCV to find the best parameters
grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("\nBest parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.4f}".format(grid_search.best_score_))

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions on test set
y_pred = best_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy on test set: {:.4f}".format(accuracy))

# Display classification report
print("\nClassification Report:")
cr = classification_report(y_test, y_pred)
print(cr)

# Feature importance
feature_importances = best_model.named_steps['rf'].feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature importances:")
print(feature_importance_df)

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title('Top 10 Feature Importances')
plt.tight_layout()
plt.show()

# Plot confusion matrix (only if number of classes is reasonable)
if len(y.unique()) <= 20:  # Only plot if 20 or fewer classes
    plt.figure(figsize=(16, 14))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', 
                xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
else:
    print("\nToo many classes to display confusion matrix clearly.")

# Function to predict career based on user input
def recommend_careers(user_responses, model, top_n=5):
    """
    Predicts the top N career recommendations based on user responses.
    
    Parameters:
    user_responses (list): List of responses to the 20 assessment questions (1-5 scale)
    model: Trained model pipeline
    top_n (int): Number of top recommendations to return
    
    Returns:
    DataFrame: Top N recommended careers with probabilities
    """
    # Convert user responses to DataFrame with the same columns as training data
    column_names = X.columns
    user_df = pd.DataFrame([user_responses], columns=column_names)
    
    # Get prediction probabilities for all classes
    proba = model.predict_proba(user_df)
    
    # Get class names (job roles)
    class_names = model.classes_
    
    # Create DataFrame with job roles and probabilities
    recommendations = pd.DataFrame({
        'Job_Role': class_names,
        'Match_Percentage': proba[0] * 100
    }).sort_values(by='Match_Percentage', ascending=False).head(top_n)
    
    return recommendations

# Example usage
print("\nExample career recommendation:")
# Example user responses to the 20 questions (on a scale of 1-5)
example_responses = [5, 3, 2, 3, 4, 5, 3, 5, 4, 4, 3, 3, 5, 3, 3, 4, 3, 5, 3, 4]
recommendations = recommend_careers(example_responses, best_model)
print(recommendations)

# Save the model
with open('career_recommendation_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
print("\nModel saved as 'career_recommendation_model.pkl'")


# Function to create personalized explanation for a career match
def explain_career_match(job_role, user_responses, feature_importance_df):
    """
    Creates a personalized explanation for why a particular job was recommended.
    
    Parameters:
    job_role (str): The recommended job role
    user_responses (list): User's responses to assessment questions
    feature_importance_df (DataFrame): Feature importance data from the model
    
    Returns:
    str: Personalized explanation
    """
    # Get the job profile from the dataset
    job_profile = df[df['Job_Role'] == job_role]
    
    # Check if the job role exists in the dataset (after filtering)
    if len(job_profile) == 0:
        return f"This career aligns with your profile based on the psychometric assessment."
    
    job_profile = job_profile.iloc[0].drop('Job_Role').values
    
    # Get top 5 important features
    top_features = feature_importance_df.head(5)['Feature'].values
    
    # Find features where user scored high (4-5) that align with job profile
    strengths = []
    for feature in top_features:
        feature_idx = list(X.columns).index(feature)
        if user_responses[feature_idx] >= 4 and job_profile[feature_idx] >= 4:
            strengths.append(feature)
    
    # Create explanation
    explanation = f"Based on your assessment, {job_role} aligns with your profile because:\n"
    
    for strength in strengths:
        explanation += f"- You scored high in {strength}, which is important for this role\n"
    
    if not strengths:
        explanation += "- Your overall pattern of responses matches professionals in this field\n"
    
    explanation += f"\nThis career typically involves strong {', '.join(top_features[:3])} abilities."
    
    return explanation

# Example explanation
print("\nExample career explanation:")
top_career = recommendations.iloc[0]['Job_Role']
explanation = explain_career_match(top_career, example_responses, feature_importance_df)
print(explanation)

# Code for the web application integration
import pickle

def get_career_recommendations_for_webapp(user_responses):
    """
    Function that would be called by the web application to get career recommendations.

    Parameters:
    user_responses (list): User's responses to the 20 assessment questions

    Returns:
    dict: Dictionary containing recommendations and explanations
    """
    # Load the trained model
    with open('D:/Mini project 2/model3/career_recommendation_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Get recommendations
    recommendations = recommend_careers(user_responses, model, top_n=5)

    # Create explanations for each recommended career
    results = []
    for _, row in recommendations.iterrows():
        job_role = row['Job_Role']
        match_percentage = row['Match_Percentage']
        explanation = explain_career_match(job_role, user_responses, feature_importance_df)

        results.append({
            'job_role': job_role,
            'match_percentage': float(match_percentage),
            'explanation': explanation
        })

    return {'recommendations': results}


# Example of how the web app would use this function
print("\nExample web app integration:")
webapp_result = get_career_recommendations_for_webapp(example_responses)
print(f"Top recommendation: {webapp_result['recommendations'][0]['job_role']}")
print(f"Match percentage: {webapp_result['recommendations'][0]['match_percentage']:.2f}%")
print("Explanation:")
print(webapp_result['recommendations'][0]['explanation'])