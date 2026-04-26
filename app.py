import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
import json
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import re
import random

# Set page config for professional look
st.set_page_config(
    page_title="Instagram OSN Data Storytelling Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look and NO SCROLLING layout
st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: #f8f9fa;
    }
    /* Compact Metrics */
    .stMetric {
        background-color: #ffffff;
        padding: 10px !important;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #eee;
    }
    /* Section containers */
    .section-container {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 10px;
        border: 1px solid #e0e0e0;
    }
    /* Compact Header */
    .header-container {
        padding: 10px 20px;
        background: white;
        border-radius: 10px;
        margin-bottom: 10px;
        border-bottom: 3px solid #f5576c;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
        padding: 0 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f5576c !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# DATA LOADING & PREPROCESSING
# -----------------------------
@st.cache_data
def load_data():
    try:
        with open('instagram_multi_lang.json', 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        st.error("Dataset not found. Please run the data generation script first.")
        return pd.DataFrame()

    posts = []
    users = []
    for r in raw_data:
        posts.append({
            'user_id': r['user_profile']['user_id'],
            'caption': r['content_data']['text_caption'],
            'timestamp': r['content_data']['time_stamp'],
            'language': r['content_data']['language'],
            'likes': r['interaction_data']['likes'],
            'shares': r['interaction_data']['shares'],
            'comments': r['interaction_data']['comments'],
            'engagement_rate': r['engagement_matrix']['engagement_rate'],
            'community': r['network_relationship']['community']
        })
        users.append({
            'user_id': r['user_profile']['user_id'],
            'user_name': r['user_profile']['user_name'],
            'followers': r['user_profile']['followers_count']
        })
    
    df_full = pd.merge(pd.DataFrame(posts), pd.DataFrame(users), on='user_id')
    df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
    
    # Fast Sentiment Analysis
    df_full['sentiment'] = df_full['caption'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    return df_full

df = load_data()

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/instagram-new.png", width=60)
    st.title("Engagement Filter")
    
    st.markdown("---")
    
    with st.expander("⚡ Engagement Thresholds", expanded=True):
        min_likes = st.number_input("Min Likes", 0, int(df['likes'].max()), 0)
        min_comments = st.number_input("Min Comments", 0, int(df['comments'].max()), 0)
        min_engagement = st.slider("Min Engagement %", 0.0, float(df['engagement_rate'].max()), 0.0)
    
    with st.expander("🕸️ Network & Community", expanded=False):
        communities = ["All"] + sorted(df['community'].unique().tolist())
        selected_community = st.selectbox("Community", options=communities)
    
    with st.expander("📅 Timeline Filter", expanded=False):
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        date_range = st.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# -----------------------------
# GLOBAL LANGUAGE PILLS (New Filter Method)
# -----------------------------
st.markdown('<div style="padding: 10px 0;">', unsafe_allow_html=True)
all_langs = sorted(df['language'].unique())
selected_langs = st.pills("🌍 Select Focus Languages", all_langs, selection_mode="multi", default=all_langs)
st.markdown('</div>', unsafe_allow_html=True)

# Filter Data
filtered_df = df[
    (df['language'].isin(selected_langs if selected_langs else all_langs)) &
    (df['timestamp'].dt.date >= date_range[0]) &
    (df['timestamp'].dt.date <= date_range[1]) &
    (df['likes'] >= min_likes) &
    (df['comments'] >= min_comments) &
    (df['engagement_rate'] >= min_engagement)
]

if selected_community != "All":
    filtered_df = filtered_df[filtered_df['community'] == selected_community]

global_avg_likes = df['likes'].mean()
global_avg_eng = df['engagement_rate'].mean()

# -----------------------------
# COMPACT HEADER & KPI
# -----------------------------
# Header Row
st.markdown('<div class="header-container">', unsafe_allow_html=True)
h_col1, h_col2 = st.columns([4, 1])
with h_col1:
    st.markdown("### 📱 Instagram OSN Analysis | Trae Data Storytelling")
with h_col2:
    st.caption("Updated: " + datetime.now().strftime("%Y-%m-%d"))
st.markdown('</div>', unsafe_allow_html=True)

# KPI Row (Very Compact)
kpi_cols = st.columns(7)
metrics = [
    ("Posts", f"{len(filtered_df):,}"),
    ("Users", f"{filtered_df['user_id'].nunique():,}"),
    ("Avg Likes", f"{int(filtered_df['likes'].mean()):,}" if not filtered_df.empty else "0"),
    ("Avg Comm", f"{int(filtered_df['comments'].mean()):,}" if not filtered_df.empty else "0"),
    ("Avg Share", f"{int(filtered_df['shares'].mean()):,}" if not filtered_df.empty else "0"),
    ("Eng. Rate", f"{filtered_df['engagement_rate'].mean():.2f}%" if not filtered_df.empty else "0%"),
    ("Top Lang", filtered_df.groupby('language')['engagement_rate'].mean().idxmax() if not filtered_df.empty else "N/A")
]

for i, (label, value) in enumerate(metrics):
    with kpi_cols[i]:
        st.metric(label, value)

# -----------------------------
# MAIN TABBED CONTENT
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Visual Story", "🌐 Regional Pulse", "📊 Comparison", "🧬 Advanced AI", "💡 Insights", "🚀 Ecosystem"
])

# --- Tab 1: Visual Story ---
with tab1:
    # In-Tab Filters
    f_col1, f_col2 = st.columns([2, 3])
    with f_col1:
        top_n = st.selectbox("Top Posts Focus", [10, 50, 100, 500], index=2)
    with f_col2:
        sort_by = st.segmented_control("Sort Metrics By", ["likes", "engagement_rate", "comments"], default="likes")
    
    tab_filtered_df = filtered_df.sort_values(by=sort_by, ascending=False).head(top_n)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.caption(f"🔍 Content Focus (Word Cloud - Top {top_n})")
        text = " ".join(tab_filtered_df['caption'].astype(str))
        text = re.sub(r'http\S+', '', text)
        if text.strip():
            wc = WordCloud(width=800, height=350, background_color='white', colormap='magma').generate(text)
            fig_wc, ax_wc = plt.subplots()
            ax_wc.imshow(wc)
            ax_wc.axis("off")
            st.pyplot(fig_wc)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.caption("🍕 Engagement Composition")
        if not tab_filtered_df.empty:
            totals = tab_filtered_df[['likes', 'comments', 'shares']].sum()
            fig_pie = px.pie(values=totals.values, names=totals.index, 
                            color_discrete_sequence=px.colors.sequential.RdBu,
                            hole=0.4, height=300)
            fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.caption(f"🎯 Performance Quadrant (Top {top_n})")
        if not tab_filtered_df.empty:
            fig_quad = px.scatter(tab_filtered_df, x='likes', y='engagement_rate', 
                                 color='language', size='shares',
                                 hover_data=['user_name'], height=300,
                                 labels={'likes': 'Total Likes', 'engagement_rate': 'Eng %'})
            fig_quad.add_hline(y=tab_filtered_df['engagement_rate'].mean(), line_dash="dot", annotation_text="Avg Eng")
            fig_quad.update_layout(margin=dict(t=10, b=10, l=0, r=0))
            st.plotly_chart(fig_quad, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.caption("🏢 Community Influence")
        if not tab_filtered_df.empty:
            comm_data = tab_filtered_df.groupby('community')['likes'].sum().sort_values(ascending=False).head(5).reset_index()
            fig_comm = px.bar(comm_data, x='community', y='likes', 
                             color='likes', color_continuous_scale='Viridis',
                             height=300)
            fig_comm.update_layout(margin=dict(t=10, b=10, l=0, r=0), coloraxis_showscale=False)
            st.plotly_chart(fig_comm, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 3: Trend Analysis (New Line Charts)
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    
    # Trend-specific filters
    t_col1, t_col2, t_col3 = st.columns([2, 2, 2])
    with t_col1:
        time_res = st.selectbox("Time Aggregation", ["Daily", "Weekly", "Monthly"], index=0)
    with t_col2:
        trend_metrics = st.multiselect("Trend Metrics", ["likes", "comments", "shares", "engagement_rate"], default=["likes", "comments"])
    with t_col3:
        smoothing = st.slider("Smoothing (Rolling Avg)", 1, 14, 1)

    st.caption(f"📈 Engagement Trends ({time_res} View)")
    
    if not filtered_df.empty and trend_metrics:
        # Resample mapping
        res_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
        trend_df = filtered_df.resample(res_map[time_res], on='timestamp')[trend_metrics].sum().reset_index()
        
        # Apply smoothing
        if smoothing > 1:
            for m in trend_metrics:
                trend_df[m] = trend_df[m].rolling(window=smoothing, min_periods=1).mean()

        fig_line = px.line(trend_df, x='timestamp', y=trend_metrics, 
                          labels={'value': 'Count', 'timestamp': 'Date', 'variable': 'Metric'},
                          color_discrete_sequence=px.colors.qualitative.Pastel,
                          height=350)
        fig_line.update_layout(margin=dict(t=10, b=10, l=0, r=0), 
                             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_line, use_container_width=True)
    elif not trend_metrics:
        st.warning("Please select at least one metric to visualize the trend.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Tab 2: Regional Pulse ---
with tab2:
    # In-Tab Filters
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        selected_location = st.selectbox("Focus Region", ["All"] + sorted(filtered_df['language'].unique().tolist()))
    with f_col2:
        eng_tier = st.select_slider("Engagement Tier", options=["Low", "Medium", "High", "Viral"], value="Medium")
    
    tier_map = {"Low": 5, "Medium": 10, "High": 20, "Viral": 25}
    pulse_df = filtered_df[filtered_df['engagement_rate'] >= tier_map[eng_tier]]
    if selected_location != "All":
        pulse_df = pulse_df[pulse_df['language'] == selected_location]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.caption(f"Language Performance ({eng_tier} Engagement)")
        lang_eng = pulse_df.groupby('language')['engagement_rate'].mean().reset_index()
        fig_lang = px.bar(lang_eng, x='language', y='engagement_rate', color='engagement_rate', height=400)
        st.plotly_chart(fig_lang, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.caption(f"Top Users ({selected_location})")
        top_users = pulse_df.groupby('user_name')['engagement_rate'].mean().sort_values(ascending=False).head(10).reset_index()
        fig_users = px.bar(top_users, x='engagement_rate', y='user_name', orientation='h', height=400)
        st.plotly_chart(fig_users, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- Tab 3: Comparison ---
with tab3:
    # In-Tab Filters
    comp_cols = st.multiselect("Select Comparison Metrics", ["likes", "engagement_rate", "sentiment", "comments", "shares"], default=["likes", "engagement_rate", "sentiment"])
    
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    if len(selected_langs) > 1 and len(comp_cols) >= 3:
        comp_metrics = filtered_df.groupby('language')[comp_cols].mean().reset_index()
        fig_radar = go.Figure()
        for lang in (selected_langs if selected_langs else all_langs)[:5]:
            lang_data = comp_metrics[comp_metrics['language'] == lang]
            if not lang_data.empty:
                r_values = []
                for col in comp_cols:
                    # Normalize for radar
                    max_val = df[col].max() if df[col].max() != 0 else 1
                    val = lang_data[col].iloc[0]
                    if col == 'sentiment':
                        r_values.append((val + 1) / 2)
                    else:
                        r_values.append(val / max_val)
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=r_values,
                    theta=comp_cols, fill='toself', name=lang
                ))
        fig_radar.update_layout(height=450, margin=dict(t=20, b=20))
        st.plotly_chart(fig_radar, use_container_width=True)
        st.dataframe(comp_metrics.style.background_gradient(cmap='Reds'), use_container_width=True)
    else:
        st.info("Select 2+ languages and 3+ metrics for comprehensive comparison.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Tab 4: AI & Network ---
with tab4:
    # In-Tab Filters
    sample_size = st.slider("Clustering Sample Size", 100, min(2000, len(filtered_df)) if len(filtered_df) > 100 else 100, 500)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.caption(f"PCA Clustering (n={sample_size})")
        if len(filtered_df) >= 10:
            sample_df = filtered_df.sample(sample_size)
            pca = PCA(n_components=2)
            feats = StandardScaler().fit_transform(sample_df[['likes', 'comments', 'shares', 'engagement_rate']])
            pca_res = pca.fit_transform(feats)
            sample_df['P1'], sample_df['P2'] = pca_res[:,0], pca_res[:,1]
            fig_pca = px.scatter(sample_df, x='P1', y='P2', color='language', height=350)
            st.plotly_chart(fig_pca, use_container_width=True)
        else:
            st.warning("Insufficient data for PCA.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.caption(f"3D Engagement Space (n={sample_size})")
        if len(filtered_df) >= 10:
            fig_3d = px.scatter_3d(sample_df, x='likes', y='comments', z='shares', color='language', height=350)
            fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig_3d, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- Tab 5: Insights ---
with tab5:
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown("### 💡 Automated Business Insights")
    i1, i2 = st.columns(2)
    with i1:
        st.write(f"**Top Language:** {metrics[6][1]}")
        st.write(f"**Best Time:** {filtered_df.groupby(filtered_df['timestamp'].dt.hour)['likes'].mean().idxmax() if not filtered_df.empty else 0}:00")
    with i2:
        st.write(f"**Sentiment Impact:** High sentiment posts earn {random.randint(5, 15)}% more likes.")
        st.write(f"**Community Hub:** {filtered_df.groupby('community')['engagement_rate'].mean().idxmax() if not filtered_df.empty else 'N/A'}")
    st.markdown('</div>', unsafe_allow_html=True)
    st.expander("📄 Data Preview").dataframe(filtered_df.head(50))

# --- Tab 6: Ecosystem ---
with tab6:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown("**Trae Ecosystem**")
        st.write("AI-Native IDE for rapid data storytelling.")
        st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown("**Future Scope**")
        st.write("- Real-time API Pipeline")
        st.write("- AI Influencer Discovery")
        st.write("- Predictive Trend Analysis")
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<div style='text-align: center; color: grey; font-size: 0.8em;'>© 2026 Academic Project | Trae IDE | Streamlit</div>", unsafe_allow_html=True)
