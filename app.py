"""
app.py
------
Smart E-Commerce Product Recommender
An AI-powered Streamlit web app that recommends products based on user
preferences and shopping behavior, with a modern, user-friendly interface.

Run locally with:
    streamlit run app.py

Author: Generated for internship / academic project submission.
"""

import io
import datetime as dt

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from model import ProductRecommender


# ==========================================================================
# PAGE CONFIGURATION
# ==========================================================================
st.set_page_config(
    page_title="Smart E-Commerce Recommender",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================================
# CUSTOM CSS — modern e-commerce theme, product cards, clean typography
# ==========================================================================
CUSTOM_CSS = """
<style>
    /* Overall app background */
    .stApp {
        background-color: #f7f8fc;
    }

    /* Headings */
    h1, h2, h3 {
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
        color: #1f2937;
    }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        padding: 28px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
    }
    .hero-banner h1 {
        color: white;
        margin-bottom: 4px;
        font-size: 2rem;
    }
    .hero-banner p {
        color: #eef2ff;
        font-size: 1.05rem;
        margin: 0;
    }

    /* Product card */
    .product-card {
        background: white;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border: 1px solid #eef0f4;
        margin-bottom: 18px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        height: 100%;
    }
    .product-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.10);
    }
    .product-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 2px;
    }
    .product-brand {
        font-size: 0.8rem;
        color: #6b7280;
        margin-bottom: 8px;
    }
    .product-category-badge {
        display: inline-block;
        background: #eef2ff;
        color: #4338ca;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 999px;
        margin-bottom: 10px;
    }
    .product-price {
        font-size: 1.25rem;
        font-weight: 800;
        color: #059669;
        margin: 6px 0;
    }
    .product-rating {
        color: #d97706;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .product-desc {
        font-size: 0.85rem;
        color: #4b5563;
        margin: 8px 0 4px 0;
        min-height: 40px;
    }
    .match-score-badge {
        display: inline-block;
        background: linear-gradient(90deg, #22c55e, #16a34a);
        color: white;
        font-weight: 700;
        font-size: 0.8rem;
        padding: 4px 12px;
        border-radius: 999px;
        margin-top: 8px;
    }

    /* Section header */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1f2937;
        margin-top: 10px;
        margin-bottom: 14px;
        border-left: 5px solid #2575fc;
        padding-left: 10px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }
    section[data-testid="stSidebar"] * {
        color: #f9fafb !important;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: white;
        border-radius: 12px;
        padding: 14px;
        border: 1px solid #eef0f4;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================================================
# DATA LOADING (cached so the CSV is only read once per session)
# ==========================================================================
@st.cache_data
def load_products(path: str = "products.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


@st.cache_resource
def load_recommender(df: pd.DataFrame) -> ProductRecommender:
    """Builds and caches the ML recommender so it isn't rebuilt on every rerun."""
    return ProductRecommender(df)


products_df = load_products()
recommender = load_recommender(products_df)


# ==========================================================================
# SESSION STATE INITIALIZATION
# ==========================================================================
if "wishlist" not in st.session_state:
    st.session_state.wishlist = []  # list of ProductIDs

if "last_recommendations" not in st.session_state:
    st.session_state.last_recommendations = pd.DataFrame()

if "browse_count" not in st.session_state:
    st.session_state.browse_count = 0


def add_to_wishlist(product_id: int):
    if product_id not in st.session_state.wishlist:
        st.session_state.wishlist.append(product_id)


def remove_from_wishlist(product_id: int):
    if product_id in st.session_state.wishlist:
        st.session_state.wishlist.remove(product_id)


# ==========================================================================
# REUSABLE UI COMPONENTS
# ==========================================================================
def render_product_card(row: pd.Series, match_label: str = None, key_prefix: str = ""):
    """Renders a single product as an attractive HTML card with a wishlist button."""
    stars = "⭐" * int(round(row["Rating"]))

    match_html = ""
    if match_label is not None:
        match_html = f'<div class="match-score-badge">🎯 {match_label}</div>'

    card_html = f"""
    <div class="product-card">
        <div class="product-category-badge">{row['Category']}</div>
        <div class="product-title">{row['ProductName']}</div>
        <div class="product-brand">by {row['Brand']}</div>
        <div class="product-price">₹{row['Price']:,.0f}</div>
        <div class="product-rating">{stars} {row['Rating']} / 5</div>
        <div class="product-desc">{row['Description']}</div>
        {match_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if row["ProductID"] in st.session_state.wishlist:
            if st.button("💔 Remove", key=f"{key_prefix}_remove_{row['ProductID']}", use_container_width=True):
                remove_from_wishlist(row["ProductID"])
                st.rerun()
        else:
            if st.button("🤍 Wishlist", key=f"{key_prefix}_add_{row['ProductID']}", use_container_width=True):
                add_to_wishlist(row["ProductID"])
                st.rerun()
    with col2:
        with st.popover("🔍 Similar", use_container_width=True):
            similar = recommender.similar_products(row["ProductID"], top_n=3)
            if similar.empty:
                st.write("No similar products found.")
            else:
                for _, srow in similar.iterrows():
                    st.markdown(
                        f"**{srow['ProductName']}** — ₹{srow['Price']:,.0f} "
                        f"⭐{srow['Rating']} · {srow['Similarity']}% similar"
                    )


def render_product_grid(df: pd.DataFrame, columns: int = 3, match_col: str = None, key_prefix: str = "grid"):
    """Renders a DataFrame of products in a responsive grid of cards."""
    if df.empty:
        st.info("No products match your criteria. Try adjusting your filters.")
        return

    rows = list(df.iterrows())
    for i in range(0, len(rows), columns):
        cols = st.columns(columns)
        for col, (_, row) in zip(cols, rows[i:i + columns]):
            with col:
                label = None
                if match_col and match_col in row:
                    label = f"{row[match_col]}% Match"
                render_product_card(row, match_label=label, key_prefix=key_prefix)


# ==========================================================================
# SIDEBAR NAVIGATION
# ==========================================================================
st.sidebar.markdown("## 🛍️ Smart Recommender")
st.sidebar.markdown("AI-powered shopping assistant")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home / Catalog",
        "🎯 AI Recommendations",
        "📊 Analytics Dashboard",
        "❤️ Wishlist",
        "ℹ️ About",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**🛒 Wishlist items:** {len(st.session_state.wishlist)}")
st.sidebar.markdown(f"**📦 Total products:** {len(products_df)}")
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · Scikit-learn · Plotly")


# ==========================================================================
# PAGE 1 — HOME / PRODUCT CATALOG (search, filter, browse)
# ==========================================================================
if page == "🏠 Home / Catalog":
    st.markdown(
        """
        <div class="hero-banner">
            <h1>🛍️ Smart E-Commerce Product Recommender</h1>
            <p>Discover products tailored to you — powered by AI, search, and smart filters.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-header">🔎 Search & Filter Products</div>', unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns([2, 1.3, 1.5, 1.2])
    with f1:
        search_term = st.text_input("Search by product name", placeholder="e.g. earbuds, shoes, cookware...")
    with f2:
        category_options = ["All"] + sorted(products_df["Category"].unique().tolist())
        selected_category = st.selectbox("Category", category_options)
    with f3:
        price_min, price_max = int(products_df["Price"].min()), int(products_df["Price"].max())
        price_range = st.slider("Price range (₹)", price_min, price_max, (price_min, price_max))
    with f4:
        min_rating = st.select_slider("Minimum rating", options=[0, 3.0, 3.5, 4.0, 4.5], value=0)

    # Apply filters
    filtered_df = products_df.copy()
    if search_term:
        filtered_df = filtered_df[filtered_df["ProductName"].str.contains(search_term, case=False, na=False)]
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df["Category"] == selected_category]
    filtered_df = filtered_df[
        (filtered_df["Price"] >= price_range[0]) & (filtered_df["Price"] <= price_range[1])
    ]
    filtered_df = filtered_df[filtered_df["Rating"] >= min_rating]

    sort_option = st.radio(
        "Sort by",
        ["Popularity", "Price: Low to High", "Price: High to Low", "Rating"],
        horizontal=True,
    )
    if sort_option == "Popularity":
        filtered_df = filtered_df.sort_values("PopularityScore", ascending=False)
    elif sort_option == "Price: Low to High":
        filtered_df = filtered_df.sort_values("Price", ascending=True)
    elif sort_option == "Price: High to Low":
        filtered_df = filtered_df.sort_values("Price", ascending=False)
    elif sort_option == "Rating":
        filtered_df = filtered_df.sort_values("Rating", ascending=False)

    st.markdown(f'<div class="section-header">🛒 Showing {len(filtered_df)} Products</div>', unsafe_allow_html=True)
    render_product_grid(filtered_df.reset_index(drop=True), columns=3, key_prefix="catalog")


# ==========================================================================
# PAGE 2 — AI RECOMMENDATIONS (preference form + KNN engine)
# ==========================================================================
elif page == "🎯 AI Recommendations":
    st.markdown(
        """
        <div class="hero-banner">
            <h1>🎯 AI-Powered Recommendations</h1>
            <p>Tell us what you're looking for, and our KNN-based engine will find your best matches.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-header">🧾 Tell Us Your Preferences</div>', unsafe_allow_html=True)

    with st.form("preference_form"):
        c1, c2 = st.columns(2)
        with c1:
            preferred_category = st.selectbox(
                "Preferred Category", sorted(products_df["Category"].unique().tolist())
            )
            preferred_rating = st.slider("Preferred Minimum Rating", 3.0, 5.0, 4.0, 0.1)
        with c2:
            budget_range = st.slider(
                "Budget Range (₹)",
                int(products_df["Price"].min()),
                int(products_df["Price"].max()),
                (500, 5000),
            )
            product_interest = st.text_input(
                "Product Interest (keyword, optional)", placeholder="e.g. wireless, organic, formal..."
            )

        top_n = st.slider("Number of recommendations", 3, 12, 6)
        submitted = st.form_submit_button("✨ Get My Recommendations", use_container_width=True)

    if submitted:
        recs = recommender.recommend_from_preferences(
            preferred_category=preferred_category,
            min_budget=budget_range[0],
            max_budget=budget_range[1],
            preferred_rating=preferred_rating,
            top_n=top_n * 2,  # over-fetch so keyword filtering still leaves enough results
        )

        # Optional keyword narrowing on top of the ML ranking
        if product_interest:
            keyword_mask = (
                recs["ProductName"].str.contains(product_interest, case=False, na=False)
                | recs["Description"].str.contains(product_interest, case=False, na=False)
            )
            narrowed = recs[keyword_mask]
            recs = narrowed if not narrowed.empty else recs  # fall back if keyword too narrow

        recs = recs.head(top_n).reset_index(drop=True)
        st.session_state.last_recommendations = recs

    if not st.session_state.last_recommendations.empty:
        recs = st.session_state.last_recommendations
        avg_match = recs["MatchScore_Final"].mean()

        m1, m2, m3 = st.columns(3)
        m1.metric("Recommended Products", len(recs))
        m2.metric("Average Match Score", f"{avg_match:.1f}%")
        m3.metric("Top Match", f"{recs['MatchScore_Final'].max():.1f}%")

        st.markdown('<div class="section-header">🏆 Top Recommended Products</div>', unsafe_allow_html=True)
        render_product_grid(recs, columns=3, match_col="MatchScore_Final", key_prefix="reco")

        # ------------------------------------------------------------
        # Download Recommendation Report
        # ------------------------------------------------------------
        st.markdown('<div class="section-header">📥 Download Your Recommendation Report</div>', unsafe_allow_html=True)

        report_lines = [
            "SMART E-COMMERCE PRODUCT RECOMMENDER",
            "AI Recommendation Report",
            f"Generated on: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            "YOUR PREFERENCES",
            f"- Preferred Category : {preferred_category}",
            f"- Budget Range       : Rs. {budget_range[0]} - Rs. {budget_range[1]}",
            f"- Minimum Rating     : {preferred_rating}",
            f"- Interest Keyword   : {product_interest if product_interest else 'N/A'}",
            "",
            "=" * 60,
            "RECOMMENDED PRODUCTS",
            "=" * 60,
            "",
        ]
        for i, r in recs.iterrows():
            report_lines.append(f"{i + 1}. {r['ProductName']} ({r['Brand']})")
            report_lines.append(f"   Category     : {r['Category']}")
            report_lines.append(f"   Price        : Rs. {r['Price']:,.0f}")
            report_lines.append(f"   Rating       : {r['Rating']} / 5")
            report_lines.append(f"   Match Score  : {r['MatchScore_Final']}%")
            report_lines.append(f"   Description  : {r['Description']}")
            report_lines.append("")

        report_text = "\n".join(report_lines)
        st.download_button(
            label="⬇️ Download Report (.txt)",
            data=report_text,
            file_name=f"recommendation_report_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    else:
        st.info("Fill in your preferences above and click **Get My Recommendations** to see AI-powered suggestions.")


# ==========================================================================
# PAGE 3 — ANALYTICS DASHBOARD (Plotly visualizations)
# ==========================================================================
elif page == "📊 Analytics Dashboard":
    st.markdown(
        """
        <div class="hero-banner">
            <h1>📊 Analytics Dashboard</h1>
            <p>Explore catalog insights: category mix, pricing trends, and customer ratings.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Products", len(products_df))
    k2.metric("Categories", products_df["Category"].nunique())
    k3.metric("Avg Price", f"₹{products_df['Price'].mean():,.0f}")
    k4.metric("Avg Rating", f"{products_df['Rating'].mean():.2f} ⭐")

    st.markdown('<div class="section-header">🗂️ Product Category Distribution</div>', unsafe_allow_html=True)
    cat_dist = recommender.category_distribution()
    col1, col2 = st.columns([1.2, 1])
    with col1:
        fig_pie = px.pie(
            cat_dist, names="Category", values="ProductCount",
            hole=0.45, color_discrete_sequence=px.colors.sequential.Blues_r,
        )
        fig_pie.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        fig_bar_cat = px.bar(
            cat_dist, x="ProductCount", y="Category", orientation="h",
            color="ProductCount", color_continuous_scale="Blues",
        )
        fig_bar_cat.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_bar_cat, use_container_width=True)

    st.markdown('<div class="section-header">💰 Price Analysis</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        price_stats = recommender.price_stats()
        fig_price = px.bar(
            price_stats, x="Category", y="AveragePrice",
            color="AveragePrice", color_continuous_scale="Greens",
            labels={"AveragePrice": "Average Price (₹)"},
        )
        st.plotly_chart(fig_price, use_container_width=True)
    with col4:
        fig_box = px.box(
            products_df, x="Category", y="Price", color="Category",
            points="all",
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown('<div class="section-header">⭐ Rating Distribution</div>', unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        fig_hist = px.histogram(
            products_df, x="Rating", nbins=12, color_discrete_sequence=["#f59e0b"],
        )
        fig_hist.update_layout(bargap=0.1)
        st.plotly_chart(fig_hist, use_container_width=True)
    with col6:
        fig_scatter = px.scatter(
            products_df, x="Price", y="Rating", color="Category", size="PopularityScore",
            hover_name="ProductName",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


# ==========================================================================
# PAGE 4 — WISHLIST
# ==========================================================================
elif page == "❤️ Wishlist":
    st.markdown(
        """
        <div class="hero-banner">
            <h1>❤️ Your Wishlist</h1>
            <p>Products you've saved during this session.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.wishlist:
        st.info("Your wishlist is empty. Browse the catalog and click **🤍 Wishlist** to save products here.")
    else:
        wishlist_df = products_df[products_df["ProductID"].isin(st.session_state.wishlist)].reset_index(drop=True)

        w1, w2 = st.columns(2)
        w1.metric("Saved Products", len(wishlist_df))
        w2.metric("Total Value", f"₹{wishlist_df['Price'].sum():,.0f}")

        st.markdown('<div class="section-header">🛍️ Saved Items</div>', unsafe_allow_html=True)
        render_product_grid(wishlist_df, columns=3, key_prefix="wishlist")

        if st.button("🗑️ Clear Wishlist", use_container_width=True):
            st.session_state.wishlist = []
            st.rerun()


# ==========================================================================
# PAGE 5 — ABOUT
# ==========================================================================
elif page == "ℹ️ About":
    st.markdown(
        """
        <div class="hero-banner">
            <h1>ℹ️ About This Project</h1>
            <p>Smart E-Commerce Product Recommender — an AI/ML intermediate-level Streamlit project.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### 🎯 Project Goal
        Build an AI-powered e-commerce web application that recommends products based on
        user preferences and shopping behavior, while delivering a modern, user-friendly
        shopping experience.

        ### 🧠 How the AI Works
        - **Preference-Based Recommendations:** Uses a **K-Nearest Neighbors (KNN)** model
          from scikit-learn. User preferences (category, budget, rating) are converted into
          a feature vector and compared against every product's feature vector to find the
          closest matches, converted into a match score (%).
        - **Similar Products:** Uses **cosine similarity** over the same feature space to
          find products that resemble a given item.

        ### 🛠️ Tech Stack
        - **Python** — core language
        - **Streamlit** — interactive web UI
        - **Pandas / NumPy** — data handling
        - **Scikit-learn** — KNN + similarity models
        - **Plotly** — interactive analytics visualizations

        ### 📁 Project Files
        | File | Purpose |
        |------|---------|
        | `app.py` | Streamlit UI, navigation, pages |
        | `model.py` | AI recommendation engine (KNN + cosine similarity) |
        | `products.csv` | Sample product dataset |
        | `requirements.txt` | Python dependencies |
        | `README.md` | Setup & usage instructions |

        ---
        *Built as an intermediate-level AI/ML + Streamlit portfolio project.*
        """
    )
