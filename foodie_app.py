"""
🍜 Foodie Club — Restaurant Recommendation App
Built with Streamlit · CSULB Mentorship Project

Run locally:
    pip install streamlit pandas scikit-learn
    streamlit run foodie_app.py

Deploy free:
    Push to GitHub → go to share.streamlit.io → connect repo → done!
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -- PAGE CONFIG -------------------------------------------------------
st.set_page_config(
    page_title="Foodie Club Picks",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- CUSTOM CSS -------------------------------------------------------
st.markdown("""
<style>
    /* Warm, food-inspired palette */
    :root {
        --primary:   #D85A30;
        --accent:    #1D9E75;
        --bg-card:   #FAFAF8;
        --text-muted:#6B7280;
    }
    .main { background-color: #FEFEFE; }
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 1.5px solid #E5E7EB;
        padding: 0.65rem 1rem;
        font-size: 1rem;
    }
    .stTextInput > div > div > input:focus {
        border-color: #D85A30;
        box-shadow: 0 0 0 3px rgba(216,90,48,0.12);
    }
    .restaurant-card {
        background: #FAFAF8;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.75rem;
        transition: box-shadow 0.2s;
    }
    .restaurant-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.07); }
    .stars { color: #F59E0B; font-size: 1.1rem; }
    .price-badge {
        background: #EAF3DE; color: #3B6D11;
        border-radius: 20px; padding: 2px 10px;
        font-size: 0.8rem; font-weight: 600;
    }
    .tag {
        background: #F1EFE8; color: #444441;
        border-radius: 20px; padding: 2px 10px;
        font-size: 0.75rem; margin-right: 4px;
    }
    .rank-badge {
        background: #D85A30; color: white;
        border-radius: 50%; width: 28px; height: 28px;
        display: inline-flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.85rem;
    }
    .header-section {
        padding: 1.5rem 0 1rem;
        border-bottom: 1px solid #E5E7EB;
        margin-bottom: 1.5rem;
    }
    .match-bar {
        background: #E1F5EE;
        height: 4px;
        border-radius: 2px;
        margin-top: 8px;
    }
    .match-bar-fill {
        background: #1D9E75;
        height: 4px;
        border-radius: 2px;
    }
    div[data-testid="metric-container"] {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)


# -- DATA LOADING -------------------------------------------------------
@st.cache_resource
def load_model_and_data():
    """Load pre-built TF-IDF model from notebook, or build from CSV."""
    if os.path.exists("tfidf_model.pkl"):
        with open("tfidf_model.pkl", "rb") as f:
            bundle = pickle.load(f)
        return bundle["vectorizer"], bundle["matrix"], bundle["df"]

    # Fallback: build from CSV if pickle not found
    if os.path.exists("restaurants_clean.csv"):
        df = pd.read_csv("restaurants_clean.csv")
    else:
        # Last-resort: hard-coded sample data so the app always works
        data = [
            {"name":"Shin-Sen-Gumi Ramen","categories":"Ramen, Japanese","city":"Gardena","stars":4.5,"review_count":1823,"price_label":"$$","price_tier":2,"has_wifi":False,"outdoor_seating":False,"good_for_kids":True,"takes_reservations":True,"vegetarian_friendly":False},
            {"name":"Guerrilla Tacos","categories":"Mexican, Tacos, Street Food","city":"Los Angeles","stars":4.3,"review_count":987,"price_label":"$","price_tier":1,"has_wifi":False,"outdoor_seating":True,"good_for_kids":False,"takes_reservations":False,"vegetarian_friendly":True},
            {"name":"Howlin' Ray's","categories":"Hot Chicken, Southern","city":"Los Angeles","stars":4.5,"review_count":2341,"price_label":"$$","price_tier":2,"has_wifi":False,"outdoor_seating":False,"good_for_kids":False,"takes_reservations":False,"vegetarian_friendly":False},
            {"name":"Providence","categories":"Seafood, Fine Dining","city":"Los Angeles","stars":4.7,"review_count":1456,"price_label":"$$$$","price_tier":4,"has_wifi":True,"outdoor_seating":False,"good_for_kids":False,"takes_reservations":True,"vegetarian_friendly":False},
            {"name":"Bavel","categories":"Middle Eastern, Mediterranean","city":"Los Angeles","stars":4.5,"review_count":1345,"price_label":"$$$","price_tier":3,"has_wifi":True,"outdoor_seating":True,"good_for_kids":True,"takes_reservations":True,"vegetarian_friendly":True},
            {"name":"Night + Market Song","categories":"Thai, Asian, Bar","city":"Los Angeles","stars":4.3,"review_count":1788,"price_label":"$$","price_tier":2,"has_wifi":True,"outdoor_seating":False,"good_for_kids":False,"takes_reservations":False,"vegetarian_friendly":True},
            {"name":"Pine & Crane","categories":"Taiwanese, Asian, Vegetarian","city":"Los Angeles","stars":4.4,"review_count":934,"price_label":"$$","price_tier":2,"has_wifi":True,"outdoor_seating":True,"good_for_kids":True,"takes_reservations":False,"vegetarian_friendly":True},
            {"name":"Sonoratown","categories":"Mexican, Tacos, Fast Food","city":"Los Angeles","stars":4.5,"review_count":1122,"price_label":"$","price_tier":1,"has_wifi":False,"outdoor_seating":True,"good_for_kids":True,"takes_reservations":False,"vegetarian_friendly":True},
            {"name":"Dama","categories":"Mexican, Fine Dining, Cocktail Bars","city":"Los Angeles","stars":4.4,"review_count":678,"price_label":"$$$","price_tier":3,"has_wifi":True,"outdoor_seating":False,"good_for_kids":False,"takes_reservations":True,"vegetarian_friendly":True},
            {"name":"Langer's Delicatessen","categories":"Delis, Sandwiches, Jewish","city":"Los Angeles","stars":4.4,"review_count":3412,"price_label":"$$","price_tier":2,"has_wifi":True,"outdoor_seating":False,"good_for_kids":True,"takes_reservations":False,"vegetarian_friendly":False},
        ]
        df = pd.DataFrame(data)
        st.info("ℹ️ Running with sample data. Run the Colab notebook first to generate restaurants_clean.csv and tfidf_model.pkl for the full dataset.")

    # Build feature strings and TF-IDF model
    price_words = {1:'cheap budget affordable',2:'moderate casual',3:'upscale nice date',4:'fine dining luxury'}

    def build_features(row):
        parts = [str(row.get("categories","")).replace(",", " ")]
        parts.append(price_words.get(int(row.get("price_tier",2)), "moderate"))
        if row.get("has_wifi"):            parts.append("wifi internet")
        if row.get("outdoor_seating"):     parts.append("outdoor patio outside")
        if row.get("good_for_kids"):       parts.append("family friendly kids")
        if row.get("takes_reservations"):  parts.append("reservations book")
        if row.get("vegetarian_friendly"): parts.append("vegetarian vegan plant based")
        stars = float(row.get("stars", 3))
        if stars >= 4.5: parts.append("excellent highly rated top")
        elif stars >= 4: parts.append("good popular")
        return " ".join(parts).lower()

    df["features"] = df.apply(build_features, axis=1)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000, sublinear_tf=True)
    matrix = vectorizer.fit_transform(df["features"])
    return vectorizer, matrix, df


vectorizer, tfidf_matrix, df = load_model_and_data()


def get_recommendations(prompt, top_n=5, min_stars=0.0, max_price=4,
                        require_outdoor=False, require_wifi=False,
                        require_veg=False, require_kids=False):
    mask = (df["stars"] >= min_stars) & (df["price_tier"] <= max_price)
    if require_outdoor: mask &= df.get("outdoor_seating", pd.Series(False, index=df.index)).astype(bool)
    if require_wifi:    mask &= df.get("has_wifi", pd.Series(False, index=df.index)).astype(bool)
    if require_veg:     mask &= df.get("vegetarian_friendly", pd.Series(False, index=df.index)).astype(bool)
    if require_kids:    mask &= df.get("good_for_kids", pd.Series(False, index=df.index)).astype(bool)

    filtered_df = df[mask]
    if filtered_df.empty:
        return pd.DataFrame()

    filtered_idx = filtered_df.index.tolist()
    prompt_vec = vectorizer.transform([prompt.lower()])
    sims = cosine_similarity(prompt_vec, tfidf_matrix[filtered_idx]).flatten()
    top_local = sims.argsort()[::-1][:top_n]
    top_global = [filtered_idx[i] for i in top_local]

    results = df.loc[top_global].copy()
    results["match_score"] = sims[top_local]
    results["match_pct"] = (results["match_score"] / results["match_score"].max() * 100).round(0).astype(int)
    return results


def render_stars(rating):
    full = int(rating)
    half = 1 if (rating - full) >= 0.5 else 0
    return "★" * full + "½" * half + "☆" * (5 - full - half)


def render_tags(row):
    tags = []
    if row.get("outdoor_seating"):      tags.append("🌿 Outdoor")
    if row.get("has_wifi"):             tags.append("📶 WiFi")
    if row.get("vegetarian_friendly"):  tags.append("🥬 Veg-friendly")
    if row.get("good_for_kids"):        tags.append("👨‍👩‍👧 Family")
    if row.get("takes_reservations"):   tags.append("📅 Reservations")
    return tags


# -- UI -------------------------------------------------------
st.markdown('<div class="header-section">', unsafe_allow_html=True)
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("## 🍜")
with col_title:
    st.markdown("## Foodie Club Picks")
    st.caption("Powered by your club's taste data · CSULB")
st.markdown("</div>", unsafe_allow_html=True)

# -- SIDEBAR FILTERS -------------------------------------------------------
with st.sidebar:
    st.markdown("### Filters")

    min_rating = st.slider("Minimum rating", 1.0, 5.0, 4.0, 0.5,
                           format="%.1f ⭐")

    price_map = {"$": 1, "$ $": 2, "$ $ $": 3, "$ $ $ $": 4}
    price_sel = st.select_slider("Max price",
                                 options=list(price_map.keys()),
                                 value="$ $ $")
    max_price = price_map[price_sel]

    top_n = st.slider("Results to show", 3, 10, 5)

    st.markdown("### Must-haves")
    req_outdoor = st.checkbox("Outdoor seating")
    req_wifi    = st.checkbox("WiFi")
    req_veg     = st.checkbox("Vegetarian-friendly")
    req_kids    = st.checkbox("Good for kids")

    st.markdown("---")
    st.markdown("**How it works**")
    st.caption("Type a prompt describing your vibe — cuisine, mood, price, occasion. The recommender ranks restaurants by how well their features match your words.")

# -- SEARCH BAR -------------------------------------------------------
st.markdown("#### What are you in the mood for?")
prompt = st.text_input(
    label="search",
    label_visibility="collapsed",
    placeholder="e.g.  'cheap ramen open late'  ·  'romantic date spot with cocktails'  ·  'family-friendly tacos'",
    value=st.session_state.get("prompt_input", ""),
    key="prompt_input",
)

example_prompts = [
    "cheap tacos cash only authentic",
    "romantic upscale date night cocktails",
    "vegetarian friendly outdoor patio",
    "family friendly affordable breakfast",
    "hot spicy street food late night",
]

st.markdown("**Try an example:**")
cols = st.columns(len(example_prompts))
for i, ex in enumerate(example_prompts):
    if cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
        st.session_state["prompt_input"] = ex
        st.rerun()

# -- RESULTS -------------------------------------------------------
if prompt and len(prompt.strip()) > 2:
    results = get_recommendations(
        prompt, top_n=top_n, min_stars=min_rating, max_price=max_price,
        require_outdoor=req_outdoor, require_wifi=req_wifi,
        require_veg=req_veg, require_kids=req_kids,
    )

    if results.empty:
        st.warning("No restaurants matched those filters. Try relaxing the rating or price filters.")
    else:
        st.markdown(f"**{len(results)} picks** for *\"{prompt}\"*")
        st.markdown("---")

        for rank, (_, row) in enumerate(results.iterrows(), 1):
            cats = str(row.get("categories","")).split(",")[:3]
            cat_str = " · ".join(c.strip() for c in cats)
            tags = render_tags(row)
            stars_str = render_stars(float(row.get("stars", 0)))
            match_pct = int(row.get("match_pct", 0))

            card_html = f"""
            <div class="restaurant-card">
              <div style="display:flex; align-items:flex-start; gap:14px;">
                <div>
                  <div style="font-weight:700; font-size:1.05rem; margin-bottom:2px;">{row['name']}</div>
                  <div style="color:#6B7280; font-size:0.85rem; margin-bottom:6px;">{cat_str} &nbsp;·&nbsp; {row.get('city','')}</div>
                  <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
                    <span class="stars">{stars_str}</span>
                    <span style="font-weight:600; font-size:0.9rem;">{row['stars']}</span>
                    <span style="color:#9CA3AF; font-size:0.85rem;">({row.get('review_count',0):,} reviews)</span>
                    <span class="price-badge">{row.get('price_label','??')}</span>
                  </div>
                  <div style="margin-top:8px;">
                    {''.join(f'<span class="tag">{t}</span>' for t in tags)}
                  </div>
                  <div style="margin-top:10px; font-size:0.8rem; color:#6B7280;">
                    Match strength &nbsp;<strong style="color:#1D9E75;">{match_pct}%</strong>
                  </div>
                  <div class="match-bar">
                    <div class="match-bar-fill" style="width:{match_pct}%;"></div>
                  </div>
                </div>
              </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

        # Map (only if lat/lon available)
        if "latitude" in results.columns and "longitude" in results.columns:
            map_df = results[["name","latitude","longitude"]].dropna()
            if not map_df.empty:
                st.markdown("#### 📍 On the map")
                map_data = map_df.rename(columns={"latitude":"lat","longitude":"lon"})
                st.map(map_data, zoom=11)

        # Summary metrics
        st.markdown("#### Quick stats")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg rating",   f"{results['stars'].mean():.1f} ⭐")
        m2.metric("Avg price",    results["price_label"].mode().iloc[0] if not results.empty else "—")
        m3.metric("Best match",   results.iloc[0]["name"])
        m4.metric("With outdoor", str(results.get("outdoor_seating", pd.Series(False)).sum()) + " spots")

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem; color:#9CA3AF;">
      <div style="font-size:3rem; margin-bottom:1rem;">🍽️</div>
      <div style="font-size:1.1rem; font-weight:500; margin-bottom:0.5rem;">What's the vibe tonight?</div>
      <div style="font-size:0.9rem;">Describe what you're craving above and we'll find the best Foodie Club picks for you.</div>
    </div>
    """, unsafe_allow_html=True)

# -- FOOTER -------------------------------------------------------
st.markdown("---")
st.caption("Built by the CSULB Foodie Club · Data from the Yelp Open Dataset · Powered by Streamlit")
