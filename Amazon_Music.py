import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

st.set_page_config(page_title="Amazon Music Clustering", layout="wide")


# 🎵 1. Title & Description

st.title("🎶 Amazon Music Clustering Dashboard")
st.markdown("""
Discover hidden patterns in Amazon Music tracks!  
This app groups songs with **similar audio characteristics** — like energy, danceability, and acousticness — using **K-Means Clustering**.
""")


# 📂 2. Load Data

df = pd.read_csv("/Users/yuvaraj/GUVI CLASSESS/Amazon Music/Final_amazon_music.csv")


# 📊 3. Cluster Summary

st.header("📊 Cluster Overview")

col1, col2 = st.columns(2)
with col1:
    st.write("### Cluster Distribution")
    st.bar_chart(df['cluster'].value_counts())

with col2:
    st.write("### Dataset Info")
    st.write(f"**Total Songs:** {len(df)}")
    st.write(f"**Number of Clusters:** {df['cluster'].nunique()}")


# 🔍 4. PCA Scatter Plot

st.header("🎨 Visualize Clusters (PCA Scatter Plot)")
features = ['danceability','energy','loudness','speechiness','acousticness',
            'instrumentalness','liveness','valence','tempo']

pca = PCA(n_components=2)
pca_data = pca.fit_transform(df[features])
pca_df = pd.DataFrame(pca_data, columns=['PC1','PC2'])
pca_df['cluster'] = df['cluster']

fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cluster', palette='tab10', s=60)
plt.title('PCA Projection of Song Clusters')
st.pyplot(fig)


# 🌈 5. Cluster Profile Heatmap

st.header("🔥 Cluster Feature Heatmap")
cluster_profile = df.groupby('cluster')[features].mean()

fig2, ax2 = plt.subplots(figsize=(10,6))
sns.heatmap(cluster_profile.T, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Average Feature Values by Cluster")
st.pyplot(fig2)


# 💡 6. Explore Songs by Cluster

st.header("🎧 Explore Songs by Cluster")
cluster_option = st.selectbox("Select a Cluster:", sorted(df['cluster'].unique()))

filtered = df[df['cluster'] == cluster_option]
st.write(f"### Songs in Cluster {cluster_option}")
st.dataframe(filtered.head(10))


# 💬 7. Cluster Insights (optional interpretation)

st.subheader("💬 Cluster Insights")
cluster_descriptions = {
    0: "🔥 High Energy & Danceability — Perfect for Party Playlists!",
    1: "🌿 Soft Acoustic & Calm Vibes — Great for Relaxation.",
    2: "🎤 High Speechiness — Rap, Hip-hop, or Talk Tracks.",
    3: "🎸 Live Feel — Concert or Performance-like Energy.",
    4: "🎶 Balanced Mix — All-rounder Songs.",
}

if cluster_option in cluster_descriptions:
    st.success(cluster_descriptions[cluster_option])
else:
    st.info("This cluster has mixed characteristics. Explore songs above!")


# 📁 8. Export Option

st.download_button(
    label="📥 Download Clustered Dataset (CSV)",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name='amazon_music_clusters.csv',
    mime='text/csv'
)
