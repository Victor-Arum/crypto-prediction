from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from sklearn.cluster import KMeans




def join_with_and(items):
    if not items:
        return ""
    elif len(items) == 1:
        return items[0]
    elif len(items) == 2:
        return f"{items[0]} and {items[1]}"
    else:
        return ", ".join(items[:-1]) + f", and {items[-1]}"



def data_scaler(data, show_table):
    # Data Scaling using PowerTransformer.

    # Initialize an empty DataFrame to hold all the data
    all_data = pd.DataFrame()

    # Loop through each unique cryptocurrency
    for crypto in data['Crypto'].unique():
        subset1 = data[data['Crypto'] == crypto]

        # Pick only close price
        subset2 = subset1["Close"].values.reshape(-1, 1)

        # Apply the PowerTransformer to stabilize variance and reduce the impact of outliers
        scaler = PowerTransformer(method='yeo-johnson')  # 'yeo-johnson' works with both positive and negative values
        scaled_data = scaler.fit_transform(subset2)

        # Create a DataFrame with the scaled data
        scaled_data_df = pd.DataFrame(data=scaled_data, columns=['Close'])

        # Concatenate the scaled data to the all_data DataFrame
        all_data = pd.concat([all_data, scaled_data_df])

    # Reset index to flatten the DataFrame
    all_data.reset_index(inplace=True, drop=True)

    # Insert the 'Crypto' and 'Date' columns back into the DataFrame
    all_data.insert(0, 'Crypto', data['Crypto'])
    all_data.insert(0, 'Date', data['Date'])

    if show_table:
        pivot_scaled_data = all_data.pivot(index='Crypto', columns='Date', values='Close')
        st.write(pivot_scaled_data)

    return all_data



def perform_PCA(all_data):
    # Feature extraction using PCA
    pivot_data = all_data.pivot(index='Crypto', columns='Date', values='Close')
    pivot_data2 = pivot_data.dropna(axis=1)

    # Perform PCA
    pca = PCA(n_components=2)
    principal_components2 = pca.fit_transform(pivot_data2)

    # Create a DataFrame with the principal components
    pca_df = pd.DataFrame(data=principal_components2, columns=['PC1', 'PC2'])

    pca_df["Crypto"] = pivot_data2.index.values

    # Reorder columns to move 'Crypto' to the beginning
    cols = pca_df.columns.tolist()
    cols = ['Crypto'] + [col for col in cols if col != 'Crypto']
    pca_df = pca_df[cols]

    return pca_df





cluster_coins_data = None
cluster_coins_cache_status = None

@st.cache_data
def generate_cluster_coins(cryptos, PC1):
    global cluster_coins_cache_status
    # Set the cache status to indicate that new data is being fetched
    cluster_coins_cache_status = "Fetching new data..."

    PCA_data = cluster_coins_data

    new_df2 = pd.DataFrame()

    X = PCA_data.iloc[:, [1, 2]].values
    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    for i in range(4):
        new_df = pd.DataFrame()

        new_df['PC1'] = X[y_kmeans == i, 0]
        new_df['PC2'] = X[y_kmeans == i, 1]
        new_df['Cluster'] = f'Cluster-{i}'

        new_df2 = pd.concat([new_df2, new_df])

    # Reset index to flatten the DataFrame
    new_df2.reset_index(inplace=True, drop=True)

    new_df3 = pd.DataFrame()

    # Loop through each unique cryptocurrency
    for crypto in PCA_data['Crypto'].unique():
        subset3 = PCA_data[PCA_data['Crypto'] == crypto]

        # print(subset3)

        # Check if there are any matching rows in new_df2 for each row in subset3
        subset4 = new_df2[(new_df2['PC1'].isin(subset3['PC1'])) & (new_df2['PC2'].isin(subset3['PC2']))]
        subset4.insert(0, 'Crypto', crypto)

        new_df3 = pd.concat([new_df3, subset4])

    # Sort by 'Cluster' in ascending order
    new_df3 = new_df3.sort_values(by=['Cluster', 'Crypto'])

    # Reset index to flatten the DataFrame
    new_df3.reset_index(inplace=True, drop=True)

    new_df3 = new_df3[['Cluster', 'Crypto', 'PC1', 'PC2']]

    return new_df3


def cluster_coins(PCA_data, show_info):
    global cluster_coins_data
    cluster_coins_data = PCA_data

    global cluster_coins_cache_status
    # Set the cache status to indicate that new data is being fetched
    cluster_coins_cache_status = None

    # Number of unique cryptocurrencies
    cryptos = sorted(PCA_data['Crypto'].unique())
    PC1 = sorted(PCA_data['PC1'].unique())

    cluster_coins = generate_cluster_coins(cryptos, PC1)

    if show_info:
        # Check cache status
        if cluster_coins_cache_status == "Fetching new data...":
            st.success("Success!!!")
        else:
            st.info("Success!!!")

        for cluster in cluster_coins["Cluster"].unique():
            subset = cluster_coins[cluster_coins["Cluster"]==cluster]
            cluster_cryptos = subset["Crypto"].values.tolist()
            st.write(f"#### The cryptocurrencies in {cluster} include:")
            st.write(join_with_and(cluster_cryptos))

    return cluster_coins






cluster_scatter_plot_data = None
cluster_scatter_plot_cache_status = None

@st.cache_data
def generate_cluster_scatter_plot(cryptos, PC1):
    global cluster_scatter_plot_cache_status
    # Set the cache status to indicate that new data is being fetched
    cluster_scatter_plot_cache_status = "Fetching new data..."

    PCA_data = cluster_scatter_plot_data

    # Extract the relevant PCA components (e.g., PC1 and PC2)
    X = PCA_data.iloc[:, [1, 2]].values

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    # Add the cluster labels to the DataFrame for Plotly
    PCA_data['Cluster'] = y_kmeans

    # Create a Plotly scatter plot
    fig = px.scatter(
        PCA_data,
        x=PCA_data.columns[1],  # PC1 (1st PCA component)
        y=PCA_data.columns[2],  # PC2 (2nd PCA component)
        color='Cluster',
        title='Clusters of Cryptocurrencies',
        labels={PCA_data.columns[1]: 'PC1', PCA_data.columns[2]: 'PC2'},
        hover_data=['Crypto'],  # Display crypto names on hover
        color_continuous_scale=px.colors.sequential.Sunset # Optional: change color scale
    )

    # Add the centroids of the clusters to the plot
    centroids = kmeans.cluster_centers_
    fig.add_scatter(
        x=centroids[:, 0],
        y=centroids[:, 1],
        mode='markers',
        marker=dict(color='yellow', size=10, symbol='x'),
        name='Centroids'
    )

    # Return the Plotly figure
    return fig


def cluster_scatter_plot(PCA_data):
    global cluster_scatter_plot_data
    cluster_scatter_plot_data = PCA_data

    global cluster_scatter_plot_cache_status
    # Set the cache status to indicate that new data is being fetched
    cluster_scatter_plot_cache_status = None

    # Number of unique cryptocurrencies
    cryptos = sorted(PCA_data['Crypto'].unique())
    PC1 = sorted(PCA_data['PC1'].unique())

    fig = generate_cluster_scatter_plot(cryptos, PC1)

    # Check cache status
    if cluster_scatter_plot_cache_status == "Fetching new data...":
        st.success("Success!!!")
    else:
        st.info("Success!!!")

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)






cluster_line_plot_clustered_data = None
cluster_line_plot_data = None
cluster_line_plot_cache_status = None

@st.cache_data
def generate_cluster_line_plot(cryptos, dates):
    global cluster_line_plot_cache_status
    # Set the cache status to indicate that new data is being fetched
    cluster_line_plot_cache_status = "Fetching new data..."

    clustered_data = cluster_line_plot_clustered_data
    data = cluster_line_plot_data
    # Dictionary to store plots
    plots = {}

    clusters = sorted(clustered_data['Cluster'].unique())
    for cluster in clusters:
        # Filter the DataFrame based on the selected cluster
        cluster_df = clustered_data[clustered_data['Cluster'] == cluster]

        # Get the unique cryptocurrencies in this cluster
        cryptos = cluster_df['Crypto'].unique()
        total = len(cryptos)

        # Determine the layout of the subplots
        cols = 3  # Adjust number of columns as needed
        rows = (total + cols - 1) // cols

        # Create subplots for Plotly
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=cryptos)

        for i, crypto in enumerate(cryptos):
            row = (i // cols) + 1
            col = (i % cols) + 1

            # Filter the data for the specific crypto
            subset = data[data['Crypto'] == crypto]

            # Add the line plot for each crypto to the subplot
            fig.add_trace(
                go.Scatter(x=subset['Date'], y=subset['Close'], mode='lines', name=crypto),
                row=row, col=col
            )

            # Update axis titles for each subplot
            fig.update_xaxes(title_text="Date", row=row, col=col)
            fig.update_yaxes(title_text="Close Price", row=row, col=col)

        # Update the layout of the entire figure
        fig.update_layout(
            height=400 * rows,  # Adjust height per row
            title_text=f"Line Plot of Cryptos in Cluster {cluster}",
            showlegend=False  # Hide legends for each subplot
        )

        # Store the Plotly figure in the dictionary
        plots[f"{cluster}_line_plot"] = fig

    return plots


def cluster_line_plot(clustered_data, data):
    global cluster_line_plot_clustered_data
    cluster_line_plot_clustered_data = clustered_data
    global cluster_line_plot_data
    cluster_line_plot_data = data

    global cluster_line_plot_cache_status
    # Set the cache status to indicate that new data is being fetched
    cluster_line_plot_cache_status = None

    # Number of unique cryptocurrencies
    cryptos = sorted(data['Crypto'].unique())
    dates = sorted(data['Date'].unique())

    plots = generate_cluster_line_plot(cryptos, dates)

    # Check cache status
    if cluster_line_plot_cache_status == "Fetching new data...":
        st.success("Success!!!")
    else:
        st.info("Success!!!")

    # number = 1
    # Iterate through the dictionary and display each plot
    for cluster_key, plot in plots.items():
        # st.write(f'##### Line Plot of Coins in Cluster-{number}')
        st.plotly_chart(plot, use_container_width=True)  # Display Plotly chart
        # number += 1


