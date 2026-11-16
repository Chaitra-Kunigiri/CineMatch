# CineMatch Streamlit application
# Run this file from the CineMatch project folder with: streamlit run 04_App/app.py

from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ------------------------------ PAGE SETTINGS ------------------------------
st.set_page_config(
    page_title="CineMatch | Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------ PROJECT PATHS ------------------------------
# Find the project root whether this file is inside 04_App or run from elsewhere.
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent if APP_DIR.name == "04_App" else APP_DIR
MODEL_PATH = PROJECT_ROOT / "03_Models" / "cinematch_artifacts.joblib"


# ------------------------------ LOAD MODEL ARTIFACTS ------------------------------
@st.cache_resource
def load_artifacts(path_string):
    """Load the final model bundle created by Notebook 08."""
    return joblib.load(path_string)


if not MODEL_PATH.exists():
    st.error("CineMatch model file was not found.")
    st.markdown(
        """
        **Expected file:**
        `03_Models/cinematch_artifacts.joblib`

        Go back to **Notebook 08 – Movie Recommendation Testing** and run
        all cells, including the final cell that saves `cinematch_artifacts.joblib`.
        """
    )
    st.stop()

try:
    artifacts = load_artifacts(str(MODEL_PATH))
except Exception as error:
    st.error(f"The model artifact could not be loaded: {error}")
    st.stop()


movies = artifacts["movies"].copy().reset_index(drop=True)
content_matrix = artifacts["content_matrix"]
content_nn = artifacts["content_nn"]
user_item = artifacts["user_item"]
item_similarity = artifacts["item_similarity"]
user_ids = artifacts["user_ids"]
rated_movie_ids = artifacts["rated_movie_ids"]
user_to_idx = artifacts["user_to_idx"]
movie_to_idx = artifacts["movie_to_idx"]
hybrid_config = artifacts.get(
    "hybrid_config",
    {"content_weight": 0.4, "collaborative_weight": 0.6},
)

# Keep IDs consistently typed for lookups.
movies["movieId"] = movies["movieId"].astype(int)
movies["title"] = movies["title"].fillna("Unknown title").astype(str)
movies["genres"] = movies["genres"].fillna("Unknown").astype(str)


# ------------------------------ RECOMMENDATION FUNCTIONS ------------------------------
def find_movie_matches(query, limit=30):
    """Return catalogue entries whose titles contain the search text."""
    query = str(query).strip().casefold()
    if not query:
        return movies.iloc[0:0]

    mask = movies["title"].str.casefold().str.contains(query, regex=False, na=False)
    return movies.loc[mask].head(limit)


def get_movie_row(movie_id):
    """Return one movie row for a MovieLens movie ID."""
    match = movies.loc[movies["movieId"] == int(movie_id)]
    if match.empty:
        return None
    return match.iloc[0]


def similar_movies(movie_id, top_n=10):
    """Recommend movies with similar metadata using the trained TF-IDF model."""
    movie_id = int(movie_id)
    matches = movies.index[movies["movieId"] == movie_id]
    if len(matches) == 0:
        return pd.DataFrame()

    row_index = int(matches[0])
    distances, indices = content_nn.kneighbors(
        content_matrix[row_index],
        n_neighbors=min(top_n + 1, len(movies)),
    )

    output = []
    for distance, candidate_index in zip(distances[0], indices[0]):
        candidate = movies.iloc[int(candidate_index)]
        candidate_id = int(candidate["movieId"])

        if candidate_id == movie_id:
            continue

        output.append(
            {
                "movieId": candidate_id,
                "title": candidate["title"],
                "genres": candidate["genres"],
                "similarity": float(1 - distance),
                "average_rating": float(candidate.get("average_rating", 0) or 0),
                "rating_count": int(candidate.get("rating_count", 0) or 0),
            }
        )
        if len(output) >= top_n:
            break

    return pd.DataFrame(output)


def get_seen_movie_ids(user_id):
    """Find the movies already rated by a known MovieLens user."""
    if int(user_id) not in user_to_idx:
        return set()

    user_index = user_to_idx[int(user_id)]
    start, end = user_item.indptr[user_index], user_item.indptr[user_index + 1]
    return {
        int(rated_movie_ids[column])
        for column in user_item.indices[start:end]
    }


def minmax_scores(score_dict):
    """Scale candidate scores to the 0–1 range."""
    if not score_dict:
        return {}

    values = np.asarray(list(score_dict.values()), dtype=float)
    low, high = values.min(), values.max()

    if np.isclose(low, high):
        return {key: 1.0 for key in score_dict}

    return {
        key: float((value - low) / (high - low))
        for key, value in score_dict.items()
    }


def content_scores_for_user(user_id, limit_pool=300):
    """Create content scores from the user's positively rated movies."""
    if int(user_id) not in user_to_idx:
        return {}

    user_index = user_to_idx[int(user_id)]
    start, end = user_item.indptr[user_index], user_item.indptr[user_index + 1]
    columns = user_item.indices[start:end]
    ratings_for_user = user_item.data[start:end]

    liked_columns = [
        column
        for column, rating in zip(columns, ratings_for_user)
        if rating >= 4.0
    ]
    if not liked_columns:
        liked_columns = list(columns)

    scores = {}
    for column in liked_columns:
        source_movie_id = int(rated_movie_ids[column])
        source_rows = movies.index[movies["movieId"] == source_movie_id]
        if len(source_rows) == 0:
            continue

        source_row = int(source_rows[0])
        distances, indices = content_nn.kneighbors(
            content_matrix[source_row],
            n_neighbors=min(limit_pool + 1, len(movies)),
        )

        for distance, candidate_index in zip(distances[0], indices[0]):
            candidate_id = int(movies.iloc[int(candidate_index)]["movieId"])
            similarity_score = float(1 - distance)
            scores[candidate_id] = max(
                scores.get(candidate_id, 0.0),
                similarity_score,
            )

    return scores


def collaborative_scores_for_user(user_id):
    """Calculate item-item collaborative scores for a known user."""
    if int(user_id) not in user_to_idx:
        return {}

    user_index = user_to_idx[int(user_id)]
    start, end = user_item.indptr[user_index], user_item.indptr[user_index + 1]
    columns = user_item.indices[start:end]
    ratings_for_user = user_item.data[start:end]

    scores = {}
    for column, rating_value in zip(columns, ratings_for_user):
        similarities = item_similarity[:, column].tocoo()

        for candidate_column, similarity in zip(
            similarities.row, similarities.data
        ):
            if similarity > 0:
                candidate_id = int(rated_movie_ids[candidate_column])
                scores[candidate_id] = scores.get(candidate_id, 0.0) + float(
                    similarity * rating_value
                )

    return scores


def hybrid_recommendations(
    user_id,
    top_n=10,
    content_weight=0.4,
    collaborative_weight=0.6,
):
    """Combine content-based and collaborative scores for a known user."""
    if content_weight < 0 or collaborative_weight < 0:
        raise ValueError("Recommendation weights cannot be negative.")

    total_weight = content_weight + collaborative_weight
    if total_weight == 0:
        raise ValueError("At least one recommendation weight must be positive.")

    content_weight /= total_weight
    collaborative_weight /= total_weight

    seen_ids = get_seen_movie_ids(user_id)
    content_scores = minmax_scores(content_scores_for_user(user_id))
    collaborative_scores = minmax_scores(
        collaborative_scores_for_user(user_id)
    )

    candidate_ids = set(content_scores) | set(collaborative_scores)
    combined_scores = {}

    for movie_id in candidate_ids - seen_ids:
        combined_scores[movie_id] = (
            content_weight * content_scores.get(movie_id, 0.0)
            + collaborative_weight
            * collaborative_scores.get(movie_id, 0.0)
        )

    # If the user has no usable recommendation history, use a popularity fallback.
    if not combined_scores:
        return popular_movies(top_n, exclude_ids=seen_ids)

    ranked = sorted(
        combined_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]

    result = pd.DataFrame(
        ranked,
        columns=["movieId", "recommendation_score"],
    )
    return result.merge(
        movies[
            [
                "movieId",
                "title",
                "genres",
                "average_rating",
                "rating_count",
            ]
        ],
        on="movieId",
        how="left",
    )


def collaborative_recommendations(user_id, top_n=10):
    """Recommend unseen movies using item-item collaborative filtering."""
    if int(user_id) not in user_to_idx:
        return pd.DataFrame()

    user_index = user_to_idx[int(user_id)]
    start, end = user_item.indptr[user_index], user_item.indptr[user_index + 1]
    seen_columns = user_item.indices[start:end]
    seen_ratings = user_item.data[start:end]

    scores = np.zeros(len(rated_movie_ids), dtype=float)
    weights = np.zeros(len(rated_movie_ids), dtype=float)

    for column, rating_value in zip(seen_columns, seen_ratings):
        similarities = item_similarity[:, column].toarray().ravel()
        scores += similarities * rating_value
        weights += np.abs(similarities)

    predicted = np.divide(
        scores,
        weights,
        out=np.zeros_like(scores),
        where=weights > 0,
    )
    predicted[seen_columns] = -np.inf

    order = np.argsort(predicted)[::-1]
    output = []

    for column in order:
        if not np.isfinite(predicted[column]) or weights[column] == 0:
            continue

        output.append(
            {
                "movieId": int(rated_movie_ids[column]),
                "predicted_score": float(predicted[column]),
            }
        )
        if len(output) >= top_n:
            break

    if not output:
        return pd.DataFrame()

    return pd.DataFrame(output).merge(
        movies[["movieId", "title", "genres", "average_rating"]],
        on="movieId",
        how="left",
    )


def popular_movies(top_n=10, exclude_ids=None):
    """Return highly rated, frequently rated movies as a cold-start fallback."""
    exclude_ids = exclude_ids or set()

    popular = movies.copy()
    popular["average_rating"] = pd.to_numeric(
        popular.get("average_rating", 0), errors="coerce"
    ).fillna(0)
    popular["rating_count"] = pd.to_numeric(
        popular.get("rating_count", 0), errors="coerce"
    ).fillna(0)

    popular = popular.loc[~popular["movieId"].isin(exclude_ids)]
    popular = popular.sort_values(
        ["rating_count", "average_rating"],
        ascending=False,
    )

    return popular[
        ["movieId", "title", "genres", "average_rating", "rating_count"]
    ].head(top_n)


def display_movie_results(dataframe, score_column=None):
    """Display recommendation results in a compact grid of movie cards."""
    if dataframe is None or dataframe.empty:
        st.info("No recommendations were found for this selection.")
        return

    for start in range(0, len(dataframe), 3):
        columns = st.columns(3)
        batch = dataframe.iloc[start : start + 3]

        for column, (_, movie) in zip(columns, batch.iterrows()):
            with column:
                st.markdown("#### 🎞️ " + str(movie.get("title", "Unknown title")))
                st.caption(str(movie.get("genres", "Genres unavailable")))

                rating = movie.get("average_rating", np.nan)
                if pd.notna(rating):
                    st.write(f"⭐ MovieLens rating: **{float(rating):.2f}/5**")

                count = movie.get("rating_count", np.nan)
                if pd.notna(count):
                    st.caption(f"Ratings: {int(count):,}")

                if score_column and score_column in movie.index:
                    st.caption(
                        f"{score_column.replace('_', ' ').title()}: "
                        f"{float(movie[score_column]):.3f}"
                    )

                st.divider()


# ------------------------------ APP DESIGN ------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0;
    }
    .subtitle {
        color: #9CA3AF;
        font-size: 1.05rem;
        margin-top: 0.2rem;
    }
    .section-gap {
        padding-top: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-title">🎬 CineMatch</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Discover your next favourite movie with intelligent recommendations.</p>',
    unsafe_allow_html=True,
)

# Sidebar navigation.
st.sidebar.title("🍿 CineMatch")
page = st.sidebar.radio(
    "Explore",
    [
        "Home",
        "Find Similar Movies",
        "Personalized Recommendations",
        "Top Rated & Popular",
        "About the Project",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Powered by MovieLens Latest Small and machine learning.")
st.sidebar.caption("Content-Based • Collaborative • Hybrid")


# ------------------------------ HOME ------------------------------
if page == "Home":
    st.subheader("Welcome to your movie discovery hub")
    st.write(
        "Search for a movie you enjoy, discover similar titles, or explore "
        "personalized recommendations generated from MovieLens user ratings."
    )

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Movies", f"{movies['movieId'].nunique():,}")
    metric2.metric("Registered rating users", f"{len(user_ids):,}")
    metric3.metric(
        "Hybrid model",
        f"{hybrid_config.get('content_weight', 0.4):.0%} / "
        f"{hybrid_config.get('collaborative_weight', 0.6):.0%}",
    )

    st.markdown("### 🔎 Find a movie")
    query = st.text_input(
        "Enter a movie title",
        placeholder="For example: Toy Story, Jumanji, Titanic...",
        key="home_movie_search",
    )

    if query.strip():
        matches = find_movie_matches(query)
        if matches.empty:
            st.warning("No matching movie was found. Try another title.")
        else:
            options = {
                f"{row.title}  |  {row.genres}": int(row.movieId)
                for row in matches.itertuples()
            }
            selected_label = st.selectbox(
                "Choose a movie",
                list(options.keys()),
                key="home_movie_choice",
            )
            if st.button("Show similar movies", type="primary"):
                selected_id = options[selected_label]
                st.session_state["home_similar_source_id"] = selected_id
                st.session_state["home_similar_results"] = similar_movies(selected_id, 6)

    if "home_similar_results" in st.session_state:
        source = get_movie_row(st.session_state.get("home_similar_source_id"))
        if source is not None:
            st.markdown(f"### Movies similar to {source['title']}")
        display_movie_results(st.session_state["home_similar_results"], score_column="similarity")

    st.markdown("### ⭐ Popular picks")
    display_movie_results(popular_movies(6))


# ------------------------------ CONTENT-BASED RECOMMENDATIONS ------------------------------
elif page == "Find Similar Movies":
    st.subheader("Find Movies Similar to Your Favourite")
    st.write(
        "Choose a movie and CineMatch will compare its genre and text-based "
        "content representation with the movie catalogue."
    )

    default_id = st.session_state.get("selected_movie_id")
    default_row = get_movie_row(default_id) if default_id is not None else None
    default_label = (
        f"{default_row['title']} (ID: {int(default_row['movieId'])})"
        if default_row is not None
        else None
    )

    search_text = st.text_input(
        "Search movie title",
        placeholder="Type at least part of a movie title...",
        key="similar_movie_search",
    )

    if search_text.strip():
        matches = find_movie_matches(search_text)
    else:
        matches = movies.iloc[0:0]

    if not matches.empty:
        labels = [
            f"{row.title}  |  {row.genres}  |  ID: {int(row.movieId)}"
            for row in matches.itertuples()
        ]
        label_to_id = {
            label: int(row.movieId)
            for label, row in zip(labels, matches.itertuples())
        }
        selected_label = st.selectbox(
            "Select a movie",
            labels,
            key="similar_movie_selection",
        )
        selected_id = label_to_id[selected_label]

        if st.button("Recommend similar movies", type="primary"):
            st.session_state["similar_results"] = similar_movies(selected_id, 10)
            st.session_state["similar_source_id"] = selected_id

    elif search_text.strip():
        st.warning("No movie title matched your search.")

    if "similar_results" in st.session_state:
        source = get_movie_row(st.session_state.get("similar_source_id"))
        if source is not None:
            st.markdown(f"### Because you selected: {source['title']}")
        display_movie_results(
            st.session_state["similar_results"],
            score_column="similarity",
        )


# ------------------------------ PERSONALIZED RECOMMENDATIONS ------------------------------
elif page == "Personalized Recommendations":
    st.subheader("Your Personalized Movie Recommendations")
    st.write(
        "This demonstration uses an existing MovieLens user profile. "
        "Choose a user ID to generate recommendations from that user's "
        "historical ratings."
    )

    valid_user_ids = sorted(int(value) for value in user_ids)
    if not valid_user_ids:
        st.error("No user profiles were found in the saved model.")
        st.stop()

    selected_user = st.selectbox(
        "Select a MovieLens user ID",
        valid_user_ids,
    )

    content_weight = st.slider(
        "Content-based contribution",
        min_value=0,
        max_value=100,
        value=int(round(float(hybrid_config.get("content_weight", 0.4)) * 100)),
        step=5,
        help="The collaborative contribution automatically becomes the remainder.",
    )
    collaborative_weight = 100 - content_weight

    st.caption(
        f"Hybrid weighting: Content {content_weight}% • "
        f"Collaborative {collaborative_weight}%"
    )

    recommendation_type = st.radio(
        "Choose recommendation method",
        [
            "Hybrid",
            "Collaborative Filtering",
        ],
        horizontal=True,
    )

    if st.button("Get my recommendations", type="primary"):
        if recommendation_type == "Hybrid":
            recommendations = hybrid_recommendations(
                selected_user,
                top_n=12,
                content_weight=content_weight,
                collaborative_weight=collaborative_weight,
            )
            score_column = "recommendation_score"
        else:
            recommendations = collaborative_recommendations(
                selected_user,
                top_n=12,
            )
            score_column = "predicted_score"

        st.session_state["personal_results"] = recommendations
        st.session_state["personal_score_column"] = score_column
        st.session_state["personal_user"] = selected_user

    if "personal_results" in st.session_state:
        if st.session_state.get("personal_user") == selected_user:
            seen_count = len(get_seen_movie_ids(selected_user))
            st.success(
                f"Recommendations for MovieLens user {selected_user}. "
                f"Already-rated movies excluded: {seen_count}."
            )
            display_movie_results(
                st.session_state["personal_results"],
                score_column=st.session_state.get("personal_score_column"),
            )


# ------------------------------ POPULAR MOVIES ------------------------------
elif page == "Top Rated & Popular":
    st.subheader("Popular Movies in the MovieLens Catalogue")
    st.write(
        "These movies are ranked using the available rating count and "
        "average-rating information in the dataset."
    )

    number_to_show = st.slider(
        "Number of movies",
        min_value=5,
        max_value=30,
        value=15,
        step=5,
    )
    display_movie_results(popular_movies(number_to_show))


# ------------------------------ ABOUT ------------------------------
elif page == "About the Project":
    st.subheader("About CineMatch")
    st.write(
        """
        **CineMatch – Intelligent Movie Recommendation System** is a machine
        learning project built using the MovieLens Latest Small dataset.

        It demonstrates three recommendation approaches:
        """
    )

    st.markdown(
        """
        - **Content-Based Filtering:** recommends movies with similar
          metadata representations using TF-IDF and cosine distance.
        - **Collaborative Filtering:** uses patterns in user–movie ratings
          and item-item similarity.
        - **Hybrid Recommendation:** combines content-based and collaborative
          scores using configurable weights.
        """
    )

    st.markdown("### Technology stack")
    st.write(
        "Python • Pandas • NumPy • Scikit-learn • SciPy • Joblib • Streamlit"
    )

    st.markdown("### Dataset")
    st.write(
        "MovieLens Latest Small — GroupLens Research, University of Minnesota."
    )

    st.markdown("### Important limitation")
    st.info(
        "The personalized page uses existing MovieLens user IDs. It is a "
        "demonstration of recommendations from historical dataset profiles; "
        "it does not create accounts or collect new ratings from visitors."
    )

    st.markdown("### Project dataset statistics")
    stat1, stat2, stat3 = st.columns(3)
    stat1.metric("Movies", f"{movies['movieId'].nunique():,}")
    stat2.metric("Genres", f"{movies['genres'].nunique():,}")
    stat3.metric("MovieLens users", f"{len(user_ids):,}")
