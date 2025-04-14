# Career Recommendation - Random Forest Model
# Clean version with display() removed and model path fixed

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from collections import Counter
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('psychometric_dataset.csv')

# Display basic info
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check class distribution
job_counts = Counter(df['Job_Role'])
print("\nJob role distribution:")
for job, count in job_counts.items():
    print(f"{job}: {count}")

# Remove single-entry jobs
single_entry_jobs = [job for job, count in job_counts.items() if count == 1]
if single_entry_jobs:
    print(f"\nRemoving {len(single_entry_jobs)} job roles with single entries")
    df = df[~df['Job_Role'].isin(single_entry_jobs)]

# Prepare data
X = df.drop('Job_Role', axis=1)
y = df['Job_Role']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Model pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=42))
])

# Hyperparameter tuning
param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [None, 15],
    'rf__min_samples_split': [2, 5],
    'rf__max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_

# Evaluation
y_pred = best_model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
with open('career_recommendation_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# Recommendation functions
def recommend_careers(user_responses, model, top_n=5):
    column_names = X.columns
    user_df = pd.DataFrame([user_responses], columns=column_names)
    proba = model.predict_proba(user_df)
    return pd.DataFrame({
        'Job_Role': model.classes_,
        'Match_Percentage': proba[0] * 100
    }).sort_values(by='Match_Percentage', ascending=False).head(top_n)

def get_career_recommendations_for_webapp(user_responses):
    with open('career_recommendation_model.pkl', 'rb') as file:
        model = pickle.load(file)
    recommendations = recommend_careers(user_responses, model)
    return {
        'recommendations': [{
            'job_role': row['Job_Role'],
            'match_percentage': float(row['Match_Percentage']),
            'explanation': f"Based on your assessment, {row['Job_Role']} is a good match."
        } for _, row in recommendations.iterrows()]
    }
