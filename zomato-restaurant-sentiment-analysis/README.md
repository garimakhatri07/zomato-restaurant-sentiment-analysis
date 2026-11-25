# Zomato Restaurant Review Sentiment Analysis

## About the Project

This project focuses on analyzing Zomato restaurant reviews and classifying customer feedback into three sentiment categories: Positive, Negative, and Neutral.

I worked with restaurant review data and restaurant information to understand customer ratings, explore different patterns, and build a machine learning model that predicts sentiment based on review text.

I also developed a Streamlit application to make the analysis and sentiment prediction easier to explore.

## Project Objectives

- Understand customer reviews and restaurant ratings.
- Combine restaurant review data with restaurant information.
- Clean and prepare the data for analysis.
- Perform Exploratory Data Analysis (EDA).
- Create useful features for further analysis.
- Build a machine learning model for sentiment classification.
- Develop an interactive Streamlit application.

## Dataset Overview

The project uses two datasets:

- **Restaurant Reviews:** Contains customer reviews, ratings, restaurant names, and other review-related information.
- **Restaurant Metadata:** Contains restaurant details such as cuisine, cost, and other information.

| Dataset | Records |
|---|---:|
| Restaurant Reviews | 10,000 |
| Restaurant Metadata | 105 |
| Merged Dataset | 10,000 |
| Final Cleaned Dataset | 9,962 |

The two datasets were merged using the restaurant name.

## Project Workflow

### 1. Data Cleaning and Preprocessing
- Checked missing values and duplicate records.
- Removed 38 completely incomplete review records.
- Converted ratings into numeric values.
- Converted restaurant cost into numeric format.
- Prepared the data for analysis and machine learning.

### 2. Exploratory Data Analysis (EDA)
Explored the dataset through different visualizations, including:
- Customer rating distribution.
- Sentiment distribution.
- Restaurant cost and average ratings.
- Cuisine-wise average ratings.
- Ratings with and without pictures.
- Review length distribution.
- Top-rated and lowest-rated restaurants.

### 3. Feature Engineering
Created and prepared features for analysis and model development:
- Converted ratings into numeric values.
- Created a `Review_Length` feature.
- Prepared sentiment categories from restaurant ratings.
- Used TF-IDF to convert text reviews into numerical features.

### 4. Sentiment Classification

The reviews were classified into three categories based on their numeric ratings:

- **Negative:** Rating less than or equal to 2.
- **Neutral:** Rating between 2.5 and 3.5.
- **Positive:** Rating greater than or equal to 4.

### 5. Machine Learning Model

I used Logistic Regression to build the sentiment classification model.

The model development process included:
- Train-test split (80:20).
- TF-IDF Vectorization.
- Logistic Regression.
- Model evaluation using accuracy, classification report, and confusion matrix.

## Model Performance

The model achieved **81.67% accuracy** on the test dataset.

The confusion matrix and classification report were also used to understand how well the model performed across the three sentiment categories.

*Note: Sentiment labels in this project were created using customer ratings. The model predicts these rating-based sentiment categories from review text.*

## Streamlit Application

An interactive Streamlit application was developed with five main sections:

1. **Overview:** Displays project summary and important metrics.
2. **Data Visualizations:** Presents the analysis through charts and graphs.
3. **Predict Sentiment:** Allows users to enter a review and predict its sentiment.
4. **Model Performance:** Displays model evaluation results.
5. **Prediction History:** Shows previous sentiment predictions.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit
- Jupyter Notebook
- Git and GitHub

## Project Structure

```text
Zomato-Sentiment-Analysis/
│
├── Dataset/
├── Models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── Screenshots/
├── 01_Zomato_Data_Analysis.ipynb
├── README.md
├── app.py
└── requirements.txt
```

## How to Run the Project

**Step 1:** Clone the repository.

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

**Step 2:** Open the project folder.

```bash
cd Zomato-Sentiment-Analysis
```

**Step 3:** Install the required libraries.

```bash
pip install -r requirements.txt
```

**Step 4:** Run the Streamlit application.

```bash
streamlit run app.py
```

## What I Learned

Through this project, I gained practical experience in data cleaning, data analysis, feature engineering, text preprocessing, machine learning, and building an interactive application using Streamlit.

It also helped me understand how customer review data can be used to identify sentiment patterns and build a basic text classification system.

---

**Project:** Zomato Restaurant Review Sentiment Analysis  
**Developed as part of:** Labmentix Data Science in AI Internship
