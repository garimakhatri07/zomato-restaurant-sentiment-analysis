import re
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ============================================================
# 1. PAGE CONFIGURATION AND PROJECT PATHS
# ============================================================
st.set_page_config(
    page_title="Zomato | Sentiment Intelligence",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Models" / "sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "Models" / "tfidf_vectorizer.pkl"
REVIEWS_PATH = BASE_DIR / "Dataset" / "Zomato Restaurant reviews.csv"
RESTAURANTS_PATH = BASE_DIR / "Dataset" / "Zomato Restaurant names and Metadata.csv"


# ============================================================
# 2. LOAD TRAINED MODEL, VECTORIZER AND DATASETS
# ============================================================
@st.cache_resource
def load_model_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Vectorizer file not found: {VECTORIZER_PATH}")

    trained_model = joblib.load(MODEL_PATH)
    trained_vectorizer = joblib.load(VECTORIZER_PATH)
    return trained_model, trained_vectorizer


@st.cache_data
def load_project_data():
    if not REVIEWS_PATH.exists():
        raise FileNotFoundError(f"Reviews CSV not found: {REVIEWS_PATH}")
    if not RESTAURANTS_PATH.exists():
        raise FileNotFoundError(f"Restaurant metadata CSV not found: {RESTAURANTS_PATH}")

    reviews = pd.read_csv(REVIEWS_PATH)
    restaurants = pd.read_csv(RESTAURANTS_PATH)
    return reviews, restaurants


try:
    model, tfidf = load_model_artifacts()
except Exception as error:
    st.error("The trained model files could not be loaded.")
    st.code(str(error))
    st.info("Check that both .pkl files are inside the project's Models folder.")
    st.stop()

try:
    reviews_df, restaurants_df = load_project_data()
    data_error = None
except Exception as error:
    reviews_df, restaurants_df = None, None
    data_error = str(error)


# ============================================================
# 3. DATA PREPARATION AND TEXT CLEANING
# ============================================================
def clean_text(text):
    """Use the same text-cleaning operations as the training notebook."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_dashboard_data(reviews, restaurants):
    """
    Recreate the relevant notebook preparation steps for dashboard charts.
    The source CSV files are not modified.
    """
    data = reviews.copy()

    # Remove rows that contain no review record information,
    # matching the cleaning rule used in the notebook.
    required_record_fields = ["Reviewer", "Review", "Rating", "Metadata", "Time"]
    available_fields = [col for col in required_record_fields if col in data.columns]
    data = data.dropna(subset=available_fields, how="all").copy()

    data["Rating_Numeric"] = pd.to_numeric(data["Rating"], errors="coerce")

    data["Sentiment"] = pd.cut(
        data["Rating_Numeric"],
        bins=[0, 2, 3.5, 5],
        labels=["Negative", "Neutral", "Positive"],
        include_lowest=True,
    )

    # Review length is useful for the EDA view; missing review text remains missing.
    data["Review_Length"] = data["Review"].fillna("").astype(str).str.len().astype("Int64")
    data.loc[data["Review"].isna(), "Review_Length"] = pd.NA

    metadata = restaurants.copy()
    metadata["Cost"] = pd.to_numeric(
        metadata["Cost"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )

    data = data.merge(
        metadata[["Name", "Cost", "Cuisines"]],
        left_on="Restaurant",
        right_on="Name",
        how="left",
        validate="many_to_one",
    )

    return data


dashboard_df = None
if data_error is None:
    try:
        dashboard_df = prepare_dashboard_data(reviews_df, restaurants_df)
    except Exception as error:
        data_error = f"Dashboard data preparation failed: {error}"


# ============================================================
# 4. CUSTOM DASHBOARD STYLING
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background: #F5F6F8;
    }

    [data-testid="stSidebar"] {
        background: #20242C;
        border-right: 1px solid #303642;
    }

    [data-testid="stSidebar"] * {
        color: #F8FAFC;
    }

    .brand-title {
        color: #FFFFFF;
        font-size: 1.42rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin-bottom: 0.15rem;
    }

    .brand-subtitle {
        color: #B8C0CC;
        font-size: 0.70rem;
        letter-spacing: 0.08em;
        margin-bottom: 1.45rem;
    }

    .hero {
        background: linear-gradient(115deg, #20242C 0%, #343B49 100%);
        padding: 1.55rem 1.8rem;
        border-radius: 18px;
        margin-bottom: 1.2rem;
        border: 1px solid #343B49;
    }

    .hero-kicker {
        color: #F2A5AA;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.68rem;
        font-weight: 750;
        margin-bottom: 0.5rem;
    }

    .hero h1 {
        color: #FFFFFF;
        font-size: 1.8rem;
        margin: 0;
        letter-spacing: -0.035em;
    }

    .hero p {
        color: #D0D5DD;
        margin: 0.45rem 0 0 0;
        font-size: 0.92rem;
    }

    .section-title {
        color: #20242C;
        font-size: 1.08rem;
        font-weight: 750;
        margin: 1.1rem 0 0.65rem 0;
    }

    .info-card {
        background: #FFFFFF;
        border: 1px solid #E7E9EE;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 3px 12px rgba(16, 24, 40, 0.035);
        min-height: 105px;
    }

    .info-label {
        color: #667085;
        font-size: 0.76rem;
        font-weight: 600;
    }

    .info-value {
        color: #20242C;
        font-size: 1.42rem;
        font-weight: 800;
        margin-top: 0.35rem;
        letter-spacing: -0.03em;
        overflow-wrap: anywhere;
    }

    .info-note {
        color: #98A2B3;
        font-size: 0.70rem;
        margin-top: 0.25rem;
    }

    .footer-note {
        color: #98A2B3;
        font-size: 0.73rem;
        text-align: center;
        padding: 1rem 0 0.3rem 0;
    }

    div.stButton > button[kind="primary"] {
        background: #CB202D;
        border: 1px solid #CB202D;
        border-radius: 10px;
        font-weight: 700;
        min-height: 2.65rem;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #A91B26;
        border-color: #A91B26;
    }

    [data-testid="stTextArea"] textarea {
        border-radius: 10px;
        background: #FFFFFF;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def page_header(kicker, title, description):
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">{kicker}</div>
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value, note=""):
    value_size = "1.02rem" if len(str(value)) > 15 else "1.42rem"
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">{label}</div>
            <div class="info-value" style="font-size:{value_size}">{value}</div>
            <div class="info-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 5. SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown(
        '<div class="brand-title">🍽️ Zomato Insights</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="brand-subtitle">RESTAURANT REVIEW INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Data Visualizations",
            "Predict Sentiment",
            "Model Performance",
            "Prediction History",
        ],
    )

    st.markdown("---")
    st.caption("Labmentix · Data Science in AI")
    st.caption("TF-IDF · Logistic Regression")


# ============================================================
# 6. OVERVIEW DASHBOARD
# ============================================================
if page == "Overview":
    page_header(
        "Sentiment intelligence dashboard",
        "Understand what customers are saying.",
        "Explore restaurant review patterns and the performance of the sentiment classification model.",
    )

    st.markdown(
        '<div class="section-title">Project at a glance</div>',
        unsafe_allow_html=True,
    )

    if dashboard_df is not None:
        total_cleaned = len(dashboard_df)
        rated_reviews = int(dashboard_df["Rating_Numeric"].notna().sum())
        modelling_reviews = int(
            (
                dashboard_df["Sentiment"].notna()
                & dashboard_df["Review"].notna()
                & dashboard_df["Review"].astype(str).str.strip().ne("")
            ).sum()
        )
    else:
        total_cleaned, rated_reviews, modelling_reviews = "—", "—", "—"

    cols = st.columns(4)
    overview_metrics = [
        ("Cleaned review records", f"{total_cleaned:,}" if isinstance(total_cleaned, int) else total_cleaned, "After removing empty records"),
        ("Reviews with numeric ratings", f"{rated_reviews:,}" if isinstance(rated_reviews, int) else rated_reviews, "Usable rating values"),
        ("Model test accuracy", "81.67%", "Held-out test set"),
        ("ML algorithm", "Logistic Regression", "TF-IDF text features"),
    ]

    for col, (label, value, note) in zip(cols, overview_metrics):
        with col:
            metric_card(label, value, note)

    if dashboard_df is not None:
        sentiment_counts = (
            dashboard_df["Sentiment"]
            .value_counts()
            .reindex(["Negative", "Neutral", "Positive"])
            .fillna(0)
            .astype(int)
        )

        left, right = st.columns([1.1, 0.9], gap="large")

        with left:
            st.markdown(
                '<div class="section-title">Sentiment distribution</div>',
                unsafe_allow_html=True,
            )
            sentiment_chart = sentiment_counts.rename_axis("Sentiment").reset_index(name="Reviews")
            st.bar_chart(
                sentiment_chart.set_index("Sentiment"),
                horizontal=True,
                color="#CB202D",
                height=250,
            )

        with right:
            st.markdown(
                '<div class="section-title">How the model works</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="info-card">
                    <b>01 · Review input</b><br>
                    <span style="color:#667085">A customer review is entered into the application.</span><br><br>
                    <b>02 · TF-IDF transformation</b><br>
                    <span style="color:#667085">The review text is converted into numerical features.</span><br><br>
                    <b>03 · Logistic Regression</b><br>
                    <span style="color:#667085">The trained classifier predicts Positive, Neutral or Negative sentiment.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption(
            "Sentiment labels are derived from ratings: 1–2 = Negative, "
            "2.5–3.5 = Neutral, and 4–5 = Positive."
        )
    else:
        st.warning("Dataset-driven overview charts are unavailable because the CSV files could not be loaded.")
        st.code(data_error)

    st.markdown(
        '<div class="section-title">Project details</div>',
        unsafe_allow_html=True,
    )
    detail_cols = st.columns(3)
    project_details = [
        ("Datasets", "Zomato restaurant reviews and restaurant metadata"),
        ("Text representation", "TF-IDF with unigrams and bigrams"),
        ("Application", "Interactive sentiment prediction built with Streamlit"),
    ]

    for col, (heading, description) in zip(detail_cols, project_details):
        with col:
            st.markdown(
                f"""
                <div class="info-card">
                    <b>{heading}</b><br>
                    <span style="color:#667085">{description}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# 7. DATA VISUALIZATIONS
# ============================================================
elif page == "Data Visualizations":
    page_header(
        "Exploratory data analysis",
        "Explore the review data.",
        "Interactive charts generated from the project's cleaned source datasets.",
    )

    if dashboard_df is None:
        st.error("Could not load the project datasets.")
        st.code(data_error)

    else:
        # Chart 1: Rating distribution
        st.markdown(
            '<div class="section-title">1. Customer rating distribution</div>',
            unsafe_allow_html=True,
        )
        rating_counts = (
            dashboard_df["Rating_Numeric"]
            .dropna()
            .value_counts()
            .sort_index()
            .rename_axis("Rating")
            .reset_index(name="Number of reviews")
        )
        st.bar_chart(
            rating_counts.set_index("Rating"),
            color="#CB202D",
            height=300,
        )

        # Chart 2: Sentiment distribution
        st.markdown(
            '<div class="section-title">2. Sentiment distribution</div>',
            unsafe_allow_html=True,
        )
        sentiment_counts = (
            dashboard_df["Sentiment"]
            .value_counts()
            .reindex(["Negative", "Neutral", "Positive"])
            .fillna(0)
            .astype(int)
            .rename_axis("Sentiment")
            .reset_index(name="Number of reviews")
        )
        st.bar_chart(
            sentiment_counts.set_index("Sentiment"),
            horizontal=True,
            color="#249B70",
            height=250,
        )

        # Chart 3: Average rating by cost category
        st.markdown(
            '<div class="section-title">3. Average rating by cost category</div>',
            unsafe_allow_html=True,
        )
        # Match the notebook: calculate one average rating per restaurant first,
        # then group restaurants into relative cost thirds.
        cost_analysis = (
            dashboard_df.dropna(subset=["Restaurant", "Rating_Numeric", "Cost"])
            .groupby("Restaurant")
            .agg(
                Average_Rating=("Rating_Numeric", "mean"),
                Cost=("Cost", "first"),
            )
            .reset_index()
        )

        if not cost_analysis.empty and cost_analysis["Cost"].nunique() >= 3:
            try:
                cost_analysis["Cost_Category"] = pd.qcut(
                    cost_analysis["Cost"],
                    q=3,
                    labels=["Low Cost", "Medium Cost", "High Cost"],
                )
                cost_summary = (
                    cost_analysis.groupby("Cost_Category", observed=False)
                    .agg(
                        Average_Rating=("Average_Rating", "mean"),
                        Restaurant_Count=("Restaurant", "count"),
                    )
                )
                cost_chart = cost_summary[["Average_Rating"]].rename(
                    columns={"Average_Rating": "Average rating"}
                )
                st.bar_chart(
                    cost_chart,
                    color="#D99B35",
                    height=280,
                )
                st.caption(
                    "Each restaurant contributes once, using its average review rating. "
                    "Cost categories are relative thirds of listed restaurant costs."
                )
            except ValueError:
                st.info("Cost values could not be divided into three distinct restaurant groups.")
        else:
            st.info("Not enough distinct restaurant cost values to compare cost categories.")

        # Chart 4: Top cuisines - full width, horizontal labels
        st.markdown(
            '<div class="section-title">4. Top cuisines by average rating</div>',
            unsafe_allow_html=True,
        )
        cuisine_data = dashboard_df.dropna(
            subset=["Cuisines", "Rating_Numeric"]
        )[["Restaurant", "Cuisines", "Rating_Numeric"]].copy()

        if not cuisine_data.empty:
            cuisine_data["Cuisine"] = cuisine_data["Cuisines"].astype(str).str.split(",")
            cuisine_data = cuisine_data.explode("Cuisine")
            cuisine_data["Cuisine"] = cuisine_data["Cuisine"].str.strip()
            cuisine_data = cuisine_data[cuisine_data["Cuisine"].ne("")]

            cuisine_summary = (
                cuisine_data.groupby("Cuisine")
                .agg(
                    Average_Rating=("Rating_Numeric", "mean"),
                    Restaurant_Count=("Restaurant", "nunique"),
                )
                .query("Restaurant_Count >= 3")
                .sort_values("Average_Rating", ascending=True)
                .tail(10)
            )

            if not cuisine_summary.empty:
                cuisine_chart = cuisine_summary[["Average_Rating"]].rename(
                    columns={"Average_Rating": "Average rating"}
                )
                st.bar_chart(
                    cuisine_chart,
                    horizontal=True,
                    color="#6477B9",
                    height=420,
                )
                st.caption(
                    "Top 10 cuisines, including only cuisines represented by at least "
                    "three restaurants. Bars are ordered from lower to higher average rating."
                )
            else:
                st.info("No cuisine groups meet the minimum restaurant count.")
        else:
            st.info("Cuisine information is unavailable.")

        # Chart 5: Ratings with and without pictures
        st.markdown(
            '<div class="section-title">5. Average rating: with vs. without pictures</div>',
            unsafe_allow_html=True,
        )
        picture_data = dashboard_df.dropna(
            subset=["Rating_Numeric"]
        ).copy()

        if "Pictures" in picture_data.columns:
            picture_data["Picture Group"] = picture_data["Pictures"].fillna(0).gt(0).map(
                {True: "With pictures", False: "Without pictures"}
            )
            picture_summary = (
                picture_data.groupby("Picture Group")["Rating_Numeric"]
                .mean()
                .rename("Average rating")
                .to_frame()
            )
            st.bar_chart(
                picture_summary,
                horizontal=True,
                color="#249B70",
                height=220,
            )
        else:
            st.info("Picture information is unavailable in the review dataset.")

        # Chart 6: Review length distribution
        st.markdown(
            '<div class="section-title">6. Review length distribution</div>',
            unsafe_allow_html=True,
        )
        length_data = dashboard_df["Review_Length"].dropna()
        if not length_data.empty:
            length_bins = pd.cut(
                length_data,
                bins=[0, 50, 100, 200, 300, 500, 1000, float("inf")],
                labels=[
                    "1–50",
                    "51–100",
                    "101–200",
                    "201–300",
                    "301–500",
                    "501–1,000",
                    "1,000+",
                ],
                include_lowest=True,
            )
            length_counts = (
                length_bins.value_counts(sort=False)
                .rename_axis("Review length (characters)")
                .reset_index(name="Number of reviews")
            )
            st.bar_chart(
                length_counts.set_index("Review length (characters)"),
                color="#D99B35",
                height=280,
            )
        else:
            st.info("Review text length could not be calculated.")

        # Restaurant-level summary for charts 7 and 8
        st.markdown(
            '<div class="section-title">7. Top and bottom restaurants by average rating</div>',
            unsafe_allow_html=True,
        )
        restaurant_summary = (
            dashboard_df.dropna(subset=["Rating_Numeric"])
            .groupby("Restaurant")
            .agg(
                Average_Rating=("Rating_Numeric", "mean"),
                Review_Count=("Rating_Numeric", "count"),
            )
            .query("Review_Count >= 20")
        )

        if not restaurant_summary.empty:
            top_bottom_col1, top_bottom_col2 = st.columns(2, gap="large")

            top_restaurants = restaurant_summary.nlargest(10, "Average_Rating").sort_values(
                "Average_Rating", ascending=True
            )
            bottom_restaurants = restaurant_summary.nsmallest(10, "Average_Rating").sort_values(
                "Average_Rating", ascending=True
            )

            with top_bottom_col1:
                st.markdown("**Top 10 restaurants**")
                st.bar_chart(
                    top_restaurants[["Average_Rating"]].rename(
                        columns={"Average_Rating": "Average rating"}
                    ),
                    horizontal=True,
                    color="#249B70",
                    height=390,
                )

            with top_bottom_col2:
                st.markdown("**Bottom 10 restaurants**")
                st.bar_chart(
                    bottom_restaurants[["Average_Rating"]].rename(
                        columns={"Average_Rating": "Average rating"}
                    ),
                    horizontal=True,
                    color="#CB202D",
                    height=390,
                )

            st.caption("Only restaurants with at least 20 numeric ratings are included.")
        else:
            st.info("Not enough restaurant rating data for the top/bottom comparison.")

        # Chart 8: Restaurant cost vs average rating
        st.markdown(
            '<div class="section-title">8. Restaurant cost vs. average rating</div>',
            unsafe_allow_html=True,
        )
        restaurant_cost = (
            dashboard_df.dropna(subset=["Rating_Numeric", "Cost"])
            .groupby("Restaurant")
            .agg(
                Average_Rating=("Rating_Numeric", "mean"),
                Review_Count=("Rating_Numeric", "count"),
                Cost=("Cost", "first"),
            )
            .reset_index()
        )

        if not restaurant_cost.empty:
            scatter_data = restaurant_cost.rename(
                columns={
                    "Cost": "Average meal cost",
                    "Average_Rating": "Average rating",
                }
            )
            st.scatter_chart(
                scatter_data,
                x="Average meal cost",
                y="Average rating",
                size="Review_Count",
                color="#CB202D",
                height=360,
            )
            st.caption(
                "Each point represents a restaurant. Point size reflects the number "
                "of numeric ratings included in its average."
            )
        else:
            st.info("Restaurant cost and rating data are unavailable.")


# ============================================================
# 8. PREDICT SENTIMENT
# ============================================================
elif page == "Predict Sentiment":
    page_header(
        "Customer review classifier",
        "Predict review sentiment.",
        "Enter a restaurant review and let the trained model classify its sentiment.",
    )

    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    review = st.text_area(
        "Restaurant review",
        placeholder="Example: The food was delicious and the service was excellent...",
        height=150,
    )

    if st.button(
        "Predict sentiment",
        type="primary",
        use_container_width=True,
    ):
        if not review.strip():
            st.warning("Please enter a review before predicting.")
        else:
            cleaned_review = clean_text(review)

            if not cleaned_review:
                st.warning("The review has no usable text after cleaning.")
            else:
                review_vector = tfidf.transform([cleaned_review])
                prediction = model.predict(review_vector)[0]

                probability_map = {}
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(review_vector)[0]
                    probability_map = {
                        str(label): float(probability)
                        for label, probability in zip(model.classes_, probabilities)
                    }

                st.markdown(
                    '<div class="section-title">Prediction result</div>',
                    unsafe_allow_html=True,
                )

                if prediction == "Positive":
                    st.success("😊 Positive sentiment")
                elif prediction == "Negative":
                    st.error("☹️ Negative sentiment")
                else:
                    st.info("😐 Neutral sentiment")

                if probability_map:
                    st.caption(
                        "Model-estimated class probabilities. These are not guarantees "
                        "that a prediction is correct."
                    )
                    probability_df = pd.DataFrame(
                        {"Estimated probability": probability_map}
                    )
                    st.bar_chart(
                        probability_df,
                        horizontal=True,
                        height=210,
                    )

                st.session_state.prediction_history.insert(
                    0,
                    {
                        "Date & time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Review": review.strip(),
                        "Predicted sentiment": prediction,
                        "Estimated probability": (
                            f"{max(probability_map.values()):.1%}"
                            if probability_map
                            else "Not available"
                        ),
                    },
                )
                st.caption("Prediction added to this session's history.")


# ============================================================
# 9. MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":
    page_header(
        "Evaluation results",
        "Understand model performance.",
        "Recorded evaluation on the held-out test set of 1,991 reviews.",
    )

    performance_cols = st.columns(4)
    performance_metrics = [
        ("Accuracy", "81.67%"),
        ("Macro precision", "0.71"),
        ("Macro recall", "0.74"),
        ("Macro F1-score", "0.72"),
    ]

    for col, (label, value) in zip(performance_cols, performance_metrics):
        with col:
            metric_card(label, value, "Held-out test set")

    st.markdown(
        '<div class="section-title">Class-wise evaluation</div>',
        unsafe_allow_html=True,
    )
    report_df = pd.DataFrame(
        {
            "Precision": [0.80, 0.39, 0.94],
            "Recall": [0.84, 0.50, 0.87],
            "F1-score": [0.82, 0.44, 0.90],
            "Support": [486, 251, 1254],
        },
        index=["Negative", "Neutral", "Positive"],
    )
    st.dataframe(
        report_df.style.format(
            {
                "Precision": "{:.2f}",
                "Recall": "{:.2f}",
                "F1-score": "{:.2f}",
            }
        ),
        use_container_width=True,
    )

    st.markdown(
        '<div class="section-title">Confusion matrix</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Rows represent actual sentiment; columns represent predicted sentiment."
    )
    confusion_df = pd.DataFrame(
        [[410, 63, 13], [69, 125, 57], [31, 132, 1091]],
        index=["Actual Negative", "Actual Neutral", "Actual Positive"],
        columns=[
            "Predicted Negative",
            "Predicted Neutral",
            "Predicted Positive",
        ],
    )
    st.dataframe(
        confusion_df,
        use_container_width=True,
    )

    st.info(
        "The model performs best on Positive reviews. Neutral reviews are harder to "
        "distinguish (F1-score 0.44). Therefore, accuracy alone does not describe "
        "performance equally across all sentiment classes."
    )


# ============================================================
# 10. PREDICTION HISTORY
# ============================================================
elif page == "Prediction History":
    page_header(
        "Recent model activity",
        "Prediction history.",
        "Review predictions made during the current Streamlit session.",
    )

    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    history = st.session_state.prediction_history

    if history:
        st.markdown(f"**Predictions this session: {len(history)}**")
        history_df = pd.DataFrame(history)
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        if st.button("Clear prediction history"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info(
            "No predictions yet. Open Predict Sentiment and submit a review "
            "to create your history."
        )
        st.caption(
            "History is stored in Streamlit session state and resets when "
            "the session restarts."
        )


# ============================================================
# 11. FOOTER
# ============================================================
st.markdown(
    '<div class="footer-note">Zomato Restaurant Review Sentiment Analysis · Labmentix Internship Project</div>',
    unsafe_allow_html=True,
)
