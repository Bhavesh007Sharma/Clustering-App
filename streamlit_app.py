import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import time
import numpy as np

# Function to remove NA/null values
def remove_na(df):
    return df.dropna()

# Function for label encoding
def label_encode_column(df, column_name):
    le = LabelEncoder()
    df[column_name] = le.fit_transform(df[column_name])
    return df

# Function to change column type
def change_column_type(df, column_name, new_type):
    df[column_name] = df[column_name].astype(new_type)
    return df

# Streamlit UI setup
st.title("Data Processing and Clustering App")

# Initialize df as None for later use
df = None

# File uploader
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Original DataFrame")
    st.dataframe(df)
    
    # Column selection
    selected_columns = st.multiselect("Select columns to keep", df.columns)
    if selected_columns:
        df = df[selected_columns]
        st.write("DataFrame after column selection")
        st.dataframe(df)
    
    # Remove NA/null values
    if st.button("Remove NA/null values"):
        df = remove_na(df)
        st.write("DataFrame after removing NA/null values")
        st.dataframe(df)
    
    # Change column type
    column_to_change = st.selectbox("Select column to change type", df.columns)
    new_type = st.selectbox("Select new type", ["int", "float", "object"])
    if st.button("Change column type"):
        df = change_column_type(df, column_to_change, new_type)
        st.write(f"DataFrame after changing {column_to_change} to {new_type}")
        st.dataframe(df)
    
    # Label encoding
    if 'object' in df.dtypes.values:  # Check if there are any object columns
        column_to_encode = st.selectbox("Select column for label encoding", df.select_dtypes(include=['object']).columns)
        if st.button("Label Encode Column"):
            df = label_encode_column(df, column_to_encode)
            st.write(f"DataFrame after label encoding {column_to_encode}")
            st.dataframe(df)

# Check if df is not None before attempting to display it
if df is not None and not df.empty:
    st.write("Processed DataFrame")
    st.dataframe(df)
    
    # Preprocess for clustering
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df.select_dtypes(include=np.number))
    
    # Clustering options
    clustering_option = st.selectbox(
        "Choose a clustering algorithm",
        ["KMeans", "DBSCAN", "MeanShift", "Spectral Clustering", "Agglomerative Clustering"]
    )
    
    if clustering_option == "KMeans":
        n_clusters = st.slider("Number of clusters", 2, 10, 3)
        model = cluster.KMeans(n_clusters=n_clusters)
        
    elif clustering_option == "DBSCAN":
        eps = st.slider("EPS", 0.1, 2.0, 0.5)
        min_samples = st.slider("Minimum samples", 1, 10, 5)
        model = cluster.DBSCAN(eps=eps, min_samples=min_samples)
        
    elif clustering_option == "MeanShift":
        quantile = st.slider("Quantile", 0.1, 0.5, 0.2)
        bandwidth = cluster.estimate_bandwidth(scaled_df, quantile=quantile)
        model = cluster.MeanShift(bandwidth=bandwidth)
        
    elif clustering_option == "Spectral Clustering":
        n_clusters = st.slider("Number of clusters (Spectral)", 2, 10, 3)
        model = cluster.SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', affinity="nearest_neighbors")
        
    elif clustering_option == "Agglomerative Clustering":
        n_clusters = st.slider("Number of clusters (Agglomerative)", 2, 10, 3)
        linkage = st.selectbox("Linkage", ["ward", "complete", "average", "single"])
        model = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    
    if st.button("Cluster"):
        start_time = time.time()
        labels = model.fit_predict(scaled_df)
        
        # Plot
        fig, ax = plt.subplots()
        unique_labels = np.unique(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for (label, color) in zip(unique_labels, colors):
            if label == -1:
                # Black used for noise.
                color = [0, 0, 0, 1]
            
            mask = labels == label
            ax.plot(scaled_df[mask, 0], scaled_df[mask, 1], 'o', markerfacecolor=tuple(color), markersize=6, label=f'Cluster {label}')
        
        ax.set_title(f"Clusters from {clustering_option}")
        plt.legend()
        st.pyplot(fig)
        
        elapsed_time = time.time() - start_time
        st.write(f"Clustering completed in {elapsed_time:.2f} seconds.")
