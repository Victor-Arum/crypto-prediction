import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import io
import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go





def eda_info(data):
    info_buffer = StringIO()
    data.info(buf=info_buffer)
    info_str = info_buffer.getvalue()
    return info_str


def combined_data_shape(data):
    data_shape = data.shape

    st.write(f"Data Shape: {data_shape[0]} Rows and {data_shape[1]} Columns")





combined_heatmap_data = None
# combined_heatmap_selectedcoins = None
Combined_heatmap_cache_status = None
@st.cache_data
def generate_combined_data_heatmap(cryptos, dates, selected_coins4):
    # global combined_data
    global Combined_heatmap_cache_status
    # Set the cache status to indicate that new data is being fetched
    Combined_heatmap_cache_status = "Fetching new data..."

    data = combined_heatmap_data
    # Pivot the data for correlation analysis
    pivot_data1 = data.pivot(index='Date', columns='Crypto', values='Close')

    # Calculate the correlation matrix
    corr_matrix = pivot_data1.corr()

    for crypto in selected_coins4:
        st.write(f"\n{crypto}:")
        corr = corr_matrix[crypto].drop(crypto)

        st.write("Top 4 positive correlations are:")
        st.write(corr.sort_values(ascending=False).head(4))

        st.write("\nTop 4 negative correlations are:")
        st.write(corr.sort_values(ascending=True).head(4))

    # Plot the correlation matrix
    fig, ax = plt.subplots(figsize=(18, 14))
    sns.heatmap(corr_matrix.loc[selected_coins4], annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix of All Selected Cryptocurrencies using their Close prices')

    # Remove white space around the plot
    plt.tight_layout()

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)  # Close the figure to free up memory
    buf.seek(0)  # Move the pointer to the beginning of the buffer

    return buf

def combined_data_heatmap(selected_coins, data, selected_coins4):
    global combined_heatmap_data

    global Combined_heatmap_cache_status
    # Set the cache status to indicate that new data is being fetched
    Combined_heatmap_cache_status = None
    # combined_heatmap_selectedcoins = selected_coins4

    if "All Coins" in selected_coins:
        combined_heatmap_data = data
        # Number of unique cryptocurrencies
        cryptos = sorted(data['Crypto'].unique())
        dates = sorted(data['Date'].unique())
    else:
        data = data[data['Crypto'].isin(selected_coins)]
        # st.write(data.shape)
        combined_heatmap_data = data
        # Number of unique cryptocurrencies
        cryptos = sorted(selected_coins)
        dates = sorted(data['Date'].unique())

    combined_heatmap = generate_combined_data_heatmap(cryptos, dates, selected_coins4)

    # Check cache status
    if Combined_heatmap_cache_status == "Fetching new data...":
        st.success("Success!!!")
    else:
        st.info("Success!!!")

    # Display the plot image in Streamlit
    st.image(combined_heatmap)





combined_linechart_data = None
Combined_linechart_cache_status = None

@st.cache_data
def generate_combined_data_linechart(cryptos, dates):
    global Combined_linechart_cache_status
    # Set the cache status to indicate that new data is being fetched
    Combined_linechart_cache_status = "Fetching new data..."

    data = combined_linechart_data
    total = len(cryptos)

    # Determine the layout of the subplots
    cols = 3  # Adjust number of columns as needed
    rows = (total + cols - 1) // cols

    # Create subplots using Plotly
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=cryptos)

    # Loop through each unique cryptocurrency and plot its closing prices
    for i, crypto in enumerate(cryptos):
        # Filter the dataset for the current cryptocurrency
        subset = data[data['Crypto'] == crypto]

        # Calculate which row and column the subplot belongs to
        row = (i // cols) + 1
        col = (i % cols) + 1

        # Add a line plot for each cryptocurrency to the subplot
        fig.add_trace(
            go.Scatter(x=subset['Date'], y=subset['Close'], mode='lines', name=crypto, line=dict(color='purple')),
            row=row, col=col
        )

        # Update axis labels for each subplot
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Close Price", row=row, col=col)

    # Update the layout of the figure (height, title, etc.)
    fig.update_layout(
        height=400 * rows,  # Adjust height based on the number of rows
        title_text="Combined Cryptocurrency Line Charts",
        showlegend=False,  # Disable legend for individual subplots
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
    )

    return fig


def combined_data_linechart(data):
    global combined_linechart_data
    combined_linechart_data = data

    global Combined_linechart_cache_status
    # Set the cache status to indicate that new data is being fetched
    Combined_linechart_cache_status = None

    # Number of unique cryptocurrencies
    cryptos = sorted(data['Crypto'].unique())
    dates = sorted(data['Date'].unique())

    # Generate the combined line chart with Plotly
    fig = generate_combined_data_linechart(cryptos, dates)

    # Check cache status
    if Combined_linechart_cache_status == "Fetching new data...":
        st.success("Success!!!")
    else:
        st.info("Success!!!")

    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)





combined_histogram_data = None
Combined_histogram_cache_status = None

@st.cache_data
def generate_combined_data_histogram(cryptos, dates):
    global Combined_histogram_cache_status
    # Set the cache status to indicate that new data is being fetched
    Combined_histogram_cache_status = "Fetching new data..."

    data = combined_histogram_data

    total = len(cryptos)

    # Determine the layout of the subplots
    cols = 5  # Adjust number of columns as needed
    rows = (total + cols - 1) // cols

    # Create subplots using Plotly
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=cryptos)

    # Loop through each unique cryptocurrency and create histogram
    for i, crypto in enumerate(cryptos):
        subset = data[data['Crypto'] == crypto]

        # Calculate which row and column the subplot belongs to
        row = (i // cols) + 1
        col = (i % cols) + 1

        # Add histogram for each cryptocurrency to the subplot
        fig.add_trace(
            go.Histogram(x=subset['Close'], nbinsx=5, name=crypto, marker_color='purple'),
            row=row, col=col
        )

        # Update axis labels for each subplot
        fig.update_xaxes(title_text="Close Price", row=row, col=col)
        fig.update_yaxes(title_text="Frequency", row=row, col=col)

    # Update the layout of the figure
    fig.update_layout(
        height=300 * rows,  # Adjust height based on the number of rows
        title_text="Combined Cryptocurrency Histograms",
        showlegend=False,  # Disable legend for individual subplots
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
    )

    return fig


def combined_data_histogram(data):
    global combined_histogram_data
    combined_histogram_data = data

    global Combined_histogram_cache_status
    # Set the cache status to indicate that new data is being fetched
    Combined_histogram_cache_status = None

    # Number of unique cryptocurrencies
    cryptos = sorted(data['Crypto'].unique())
    dates = sorted(data['Date'].unique())

    # Generate the combined histogram chart with Plotly
    fig = generate_combined_data_histogram(cryptos, dates)

    # Check cache status
    if Combined_histogram_cache_status == "Fetching new data...":
        st.success("Success!!!")
    else:
        st.info("Success!!!")

    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)






combined_boxplot_data = None
Combined_boxplot_cache_status = None

@st.cache_data
def generate_combined_data_boxplot(cryptos, dates):
    global Combined_boxplot_cache_status
    # Set the cache status to indicate that new data is being fetched
    Combined_boxplot_cache_status = "Fetching new data..."

    data = combined_boxplot_data
    total = len(cryptos)

    # Determine the layout of the subplots
    cols = 5  # Adjust number of columns as needed
    rows = (total + cols - 1) // cols

    # Create subplots using Plotly
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=cryptos)

    # Loop through each unique cryptocurrency and create box plots
    for i, crypto in enumerate(cryptos):
        subset = data[data['Crypto'] == crypto]

        # Determine which row and column the subplot belongs to
        row = (i // cols) + 1
        col = (i % cols) + 1

        # Add box plot for each cryptocurrency to the subplot
        fig.add_trace(
            go.Box(y=subset['Close'], name=crypto, boxmean='sd'),
            row=row, col=col
        )

        # Update axis labels for each subplot
        fig.update_xaxes(title_text="Close Price", row=row, col=col)

    # Update layout of the entire figure
    fig.update_layout(
        height=300 * rows,  # Adjust height based on the number of rows
        title_text="Combined Cryptocurrency Box Plots",
        showlegend=False,  # Hide legend for individual subplots
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
    )

    return fig


def combined_data_boxplot(data):
    global combined_boxplot_data
    combined_boxplot_data = data

    global Combined_boxplot_cache_status
    # Set the cache status to indicate that new data is being fetched
    Combined_boxplot_cache_status = None

    # Number of unique cryptocurrencies
    cryptos = sorted(data['Crypto'].unique())
    dates = sorted(data['Date'].unique())

    # Generate the combined box plot with Plotly
    fig = generate_combined_data_boxplot(cryptos, dates)

    # Check cache status
    if Combined_boxplot_cache_status == "Fetching new data...":
        st.success("Success!!!")
    else:
        st.info("Success!!!")

    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

