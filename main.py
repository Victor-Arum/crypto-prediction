import streamlit as st
import traceback
from download_module import crypto_list, download_data, clean_data, data_dictionary
from eda_module import eda_info, combined_data_heatmap, combined_data_shape, combined_data_linechart, combined_data_histogram, combined_data_boxplot
from clustering_module import data_scaler, perform_PCA, cluster_coins, cluster_scatter_plot, cluster_line_plot
from prediction_module import predict_with_prophet, predict_with_arima, predict_with_lstm, predict_with_svm, forecast_with_arima, forecast_with_lstm, forecast_with_svm, forecast_with_prophet
from PIL import Image
import os




# Initialize session state for selected buttons
if 'old_button_status' not in st.session_state:
    st.session_state.old_button_status = {}

new_button_status = {}

# Initialize session state for selected buttons
if 'old_checkbox_status' not in st.session_state:
    st.session_state.old_checkbox_status = {}

new_checkbox_status = {}


# Initialize session state for downloaded coins
if 'downloaded_coins_list' not in st.session_state:
    st.session_state['downloaded_coins_list'] = []

col8, col9, col10 = st.columns([3, 1, 1])
with col8:
    # Initialize session state for Clear Dashboard Output button
    if st.button("Clear Screen" , key="clear_dashboard1"):
        st.session_state.old_button_status = { "load_data_button": False,
                                                "load_eda_button": False,
                                               "load_preprocess_button": False,
                                                "load_prediction_button": False,
                                                "load_all_button": False }
with col10:
    # Clear Cache button
    if st.button("Clear Cache", key="Clear_Cache_button"):
        st.cache_data.clear()
        st.success("Cache cleared successfully.")



# Streamlit app setup
# st.title("Cryptocurrency Dashboard")
st.markdown("<h1 style='text-align: center;'>SOLiGence Intelligence Coin Trading (ICT) Platform</h1>", unsafe_allow_html=True)

button_keys = ["load_data_button", "load_eda_button", "load_preprocess_button", "load_prediction_button"]
button_names = ["Download Data", "Load EDA", "Cluster Coins", "Predict & Forecast"]

cols = st.columns(len(button_names))  # Create a row with `num_columns` columns

for j, (button_key, button_name) in enumerate(zip(button_keys, button_names)):
    with cols[j]:  # Place each checkbox inside its respective column
        new_button_status[button_key] = st.button(button_name, key=f"button_list{j}")


new_button_status["load_all_button"] = False



cache_status = None
# Wrap the imported function in another function and apply the caching decorator
@st.cache_data
def cached_download_crypto_data(cryptos, duration):
    global cache_status
    # Set the cache status to indicate that new data is being fetched
    cache_status = "Fetching new data..."
    # Simulate data download
    data = download_data(cryptos, duration)
    return data

def join_with_and(items):
    if not items:
        return ""
    elif len(items) == 1:
        return items[0]
    elif len(items) == 2:
        return f"{items[0]} and {items[1]}"
    else:
        return ", ".join(items[:-1]) + f", and {items[-1]}"


# Initialize session state for load_data_custom_multiselect widget
if 'selected_coins1' not in st.session_state:
    st.session_state.selected_coins1 = ["Select All"]
if 'selected_coins_list1' not in st.session_state:
    st.session_state.selected_coins_list1 = []
if 'option_code1' not in st.session_state:
    st.session_state.option_code1 = 0
if 'loop_iteration1' not in st.session_state:
    st.session_state.loop_iteration1 = 0

# Initialize session state for EDA_custom_multiselect widget
if 'selected_coins2' not in st.session_state:
    st.session_state.selected_coins2 = ["All Coins"]
if 'selected_coins_list2' not in st.session_state:
    st.session_state.selected_coins_list2 = None
if 'option_code2' not in st.session_state:
    st.session_state.option_code2 = 0
if 'loop_iteration2' not in st.session_state:
    st.session_state.loop_iteration2 = 0

# Initialize session state for preprocess_custom_multiselect widget
if 'selected_coins3' not in st.session_state:
    st.session_state.selected_coins3 = ["All Coins"]
if 'selected_coins_list3' not in st.session_state:
    st.session_state.selected_coins_list3 = None
if 'option_code3' not in st.session_state:
    st.session_state.option_code3 = 0
if 'loop_iteration3' not in st.session_state:
    st.session_state.loop_iteration3 = 0

# Initialize session state for predict_custom_multiselect widget
if 'selected_coins4' not in st.session_state:
    st.session_state.selected_coins4 = []
if 'selected_coins_list4' not in st.session_state:
    st.session_state.selected_coins_list4 = None
if 'option_code4' not in st.session_state:
    st.session_state.option_code4 = 0
if 'loop_iteration4' not in st.session_state:
    st.session_state.loop_iteration4 = 0
if 'target_default4' not in st.session_state:
    st.session_state.target_default4 = ["ATOM-USD", "BTC-USD", "FIL-USD", "ZEC-USD"]

# Initialize session state for predict_custom_multiselect_b widget
if 'selected_coins4b' not in st.session_state:
    st.session_state.selected_coins4b = ["All Models"]
if 'selected_coins_list4b' not in st.session_state:
    st.session_state.selected_coins_list4b = None
if 'option_code4b' not in st.session_state:
    st.session_state.option_code4b = 0
if 'loop_iteration4b' not in st.session_state:
    st.session_state.loop_iteration4b = 0


# Use to handle button click status
if st.session_state.old_button_status == {}:
    # st.write("old_button_status is empty")
    st.session_state.old_button_status = new_button_status
    # st.write(new_button_status)

else:
    for key, value in new_button_status.items():
        if value:
            # st.write("Some button clicked")
            st.session_state.old_button_status = new_button_status


button_actions = []
for key, value in st.session_state.old_button_status.items():
    if value:
        button_actions.append("Button is Active")


if not button_actions:
    # Get the current directory of the script
    current_dir = os.path.dirname(__file__)
    # Define the image path (assuming the image is in the same directory)
    image_path = os.path.join(current_dir, 'image.jpg')
    # Open and display the image using streamlit with column width
    image = Image.open(image_path)
    st.image(image, caption='Cryptocurrency Dashboard', use_column_width=True)


# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state['data'] = {}



if st.session_state.old_button_status["load_all_button"] or st.session_state.old_button_status["load_data_button"] or st.session_state.old_button_status["load_eda_button"] or st.session_state.old_button_status["load_preprocess_button"] or st.session_state.old_button_status["load_prediction_button"]:
    st.write("---")
    try:
        if st.session_state.old_button_status["load_all_button"] or st.session_state.old_button_status["load_data_button"]:
            st.write("Please select all coins to be downloaded.")

            # Create a checkbox to select/deselect all
            select_all = st.checkbox("Select All Coins", value=True, key="coins_box0")

            # Store the state of individual checkboxes in a dictionary
            checkbox_states = {}

            # Define the number of columns per row (5 in this case)
            num_columns = 5

            # Create rows with columns
            for i in range(0, len(crypto_list), num_columns):
                cols = st.columns(num_columns)  # Create a row with `num_columns` columns
                for j, coin in enumerate(crypto_list[i:i + num_columns]):
                    with cols[j]:  # Place each checkbox inside its respective column
                        if select_all:
                            checkbox_states[coin] = st.checkbox(coin, value=True, key=f"coins_box{i}_{j}")
                        else:
                            checkbox_states[coin] = st.checkbox(coin, key=f"coins_box{i}_{j}")

            # Display selected coins
            selected_coins = [coin for coin, selected in checkbox_states.items() if selected]
            if selected_coins:
                st.write("Your selections are:")
                st.write(", ".join(selected_coins))
            else:
                st.write("You have not selected any coin!")

            st.write("")
            download_duration = st.number_input("Enter Number of Days:", min_value=1, max_value=1200, value=365, key="coins_num_input")


            if selected_coins:
                st.write("---")
                # Download data
                data = cached_download_crypto_data(selected_coins, download_duration)
                data_dict = data_dictionary(data)

                downloaded_coins_list = list(data_dict.keys())
                st.session_state['downloaded_coins_list'] = [x for x in downloaded_coins_list if x != "All Coins"]
                # st.write(st.session_state['downloaded_coins_list'])

                # Check cache status
                if cache_status == "Fetching new data...":
                    st.success("Success:---Crypto data is downloaded successfully.")
                else:
                    st.success("Success:---Crypto data is loaded successfully.")

                # Perform data cleaning
                clean_data = clean_data(data)
                clean_data_dict = data_dictionary(clean_data)

                st.write(clean_data)

                st.session_state.data = {}
                st.session_state.data["Cleaned Data"] = clean_data_dict

            else:
                st.info(f'Info:---Please select at least one coin to continue.')

        if st.session_state.old_button_status["load_all_button"] or st.session_state.old_button_status["load_eda_button"]:
            st.write("Please select all or any EDA tasks to be performed.")

            eda_tasks = ["Info", "Describe", "Shape", "Line Chart", "Histogram", "Box Plot"]

            # Create a checkbox to select/deselect all
            new_checkbox_status["eda_select_all"] = st.checkbox("Select All", key=f"eda_task_00")

            # Store the state of individual checkboxes in a dictionary
            eda_checkbox_states = {}

            # Define the number of columns per row (5 in this case)
            num_columns = 4

            # Create rows with columns
            for i in range(0, len(eda_tasks), num_columns):
                cols = st.columns(num_columns)  # Create a row with `num_columns` columns
                for j, task in enumerate(eda_tasks[i:i + num_columns]):
                    with cols[j]:  # Place each checkbox inside its respective column
                        if new_checkbox_status["eda_select_all"]:
                            eda_checkbox_states[task] = st.checkbox(task, value=True, key=f"eda_task{i}_{j}")
                        else:
                            eda_checkbox_states[task] = st.checkbox(task, key=f"eda_task{i}_{j}")


            # Display selected coins
            selected_eda_tasks = [task for task, selected in eda_checkbox_states.items() if selected]
            if selected_eda_tasks:
                st.write("You selected:")
                st.write(join_with_and(selected_eda_tasks))
            else:
                st.write("No tasks selected.")


            if selected_eda_tasks:
                st.write("---")

                if st.session_state.data == {}:
                    st.error('Error:---Please download crypto data first!!!')
                else:
                    selected_coins2 = st.session_state['downloaded_coins_list']
                    data = st.session_state.data["Cleaned Data"]["All Coins"]
                    data = data[(data['Crypto'].isin(sorted(selected_coins2)))]
                    # Reset index to flatten the DataFrame
                    data.reset_index(inplace=True, drop=True)
                    text = f"EDA for Selected Cryptocurrencies:"

                    st.write(f"## {text}")

                    for task in selected_eda_tasks:
                        st.write(f"#### {task}")
                        if task == "Describe":
                            # st.write(result)
                            st.write(data.describe())
                        elif task == "Info":
                            # st.text(result)
                            st.text(eda_info(data))
                        elif task == "Shape":
                            # st.write(result)
                            combined_data_shape(data)
                        elif task == "Line Chart":
                            combined_data_linechart(data)
                        elif task == "Histogram":
                            combined_data_histogram(data)
                        elif task == "Box Plot":
                            combined_data_boxplot(data)



            else:
                st.info(f'Info:---Please select at least 1 EDA task to proceed.')

        if st.session_state.old_button_status["load_all_button"] or st.session_state.old_button_status["load_preprocess_button"]:
            st.write("Please select all or any clustering tasks to be performed.")

            preprocess_tasks = ["Scale Data", "Perform PCA", "Cluster Coins", "Scatter Plot", "LinePlot of Clusters"]

            # Create a checkbox to select/deselect all
            new_checkbox_status["preprocess_select_all"] = st.checkbox("Select All", key="preprocess_task_00")

            # Store the state of individual checkboxes in a dictionary
            preprocess_checkbox_states = {}

            # Define the number of columns per row (5 in this case)
            num_columns = 4

            # Create rows with columns
            for i in range(0, len(preprocess_tasks), num_columns):
                cols = st.columns(num_columns)  # Create a row with `num_columns` columns
                for j, task in enumerate(preprocess_tasks[i:i + num_columns]):
                    with cols[j]:  # Place each checkbox inside its respective column
                        if new_checkbox_status["preprocess_select_all"]:
                            preprocess_checkbox_states[task] = st.checkbox(task, value=True, key=f"preprocess_task{i}_{j}")
                        else:
                            preprocess_checkbox_states[task] = st.checkbox(task, key=f"preprocess_task{i}_{j}")

            # Display selected coins
            selected_preprocess_tasks = [task for task, selected in preprocess_checkbox_states.items() if selected]
            if selected_preprocess_tasks:
                st.write("You selected:")
                st.write(join_with_and(selected_preprocess_tasks))
            else:
                st.write("No tasks selected.")



            if selected_preprocess_tasks:
                st.write("---")

                if st.session_state.data == {}:
                    st.error('Error:---Please download crypto data first!!!')
                else:
                    selected_coins3 = st.session_state['downloaded_coins_list']
                    data = st.session_state.data["Cleaned Data"]["All Coins"]
                    data = data[(data['Crypto'].isin(sorted(selected_coins3)))]
                    # Reset index to flatten the DataFrame
                    data.reset_index(inplace=True, drop=True)
                    text = f"Clustering of Selected Cryptocurrencies:"
                    text2 = "Selected Coins"

                    st.write(f"## {text}")

                    for task in selected_preprocess_tasks:

                        if task == "Scale Data":
                            st.write(f'##### Pivoted Scaled "Close" Prices for {text2}.')
                            scaled_data = data_scaler(data, show_table=True)
                            # st.write(scaled_data)

                        elif task == "Perform PCA":
                            if "Scale Data" not in selected_preprocess_tasks:
                                scaled_data = data_scaler(data, show_table=False)

                            st.write(f'##### Principal Components of {text2} using their "Close" Prices.')
                            PCA_data = perform_PCA(scaled_data)
                            st.write(PCA_data)

                        elif task == "Cluster Coins":
                            if "Perform PCA" not in selected_preprocess_tasks:
                                scaled_data = data_scaler(data, show_table=False)
                                PCA_data = perform_PCA(scaled_data)

                            st.write(f'##### Clustered Data for {text2}')
                            clustered_data = cluster_coins(PCA_data, show_info=True)
                            # st.write(clustered_data)

                        elif task == "Scatter Plot":
                            if "Perform PCA" not in selected_preprocess_tasks:
                                scaled_data = data_scaler(data, show_table=False)
                                PCA_data = perform_PCA(scaled_data)

                            st.write(f'##### Clustered Scatter Plot for {text2}')
                            cluster_scatter_plot(PCA_data)

                        elif task == "LinePlot of Clusters":
                            if "Cluster Coins" not in selected_preprocess_tasks:
                                scaled_data = data_scaler(data, show_table=False)
                                PCA_data = perform_PCA(scaled_data)
                                clustered_data = cluster_coins(PCA_data, show_info=False)

                            st.write(f'##### Clustered Line Plot for {text2}')
                            cluster_line_plot(clustered_data, data)


            else:
                st.info(f'Info:---Please select at least 1 clustering task to proceed.')

        if st.session_state.old_button_status["load_all_button"] or st.session_state.old_button_status["load_prediction_button"]:

            prediction_tasks = ["Show Heatmap", "Predict", "Forecast"]
            prediction_models = ['SVM', 'Prophet', 'ARIMA', 'LSTM']
            prediction_coins = ['AVAX-USD', 'BNB-USD', 'MATIC-USD', 'XMR-USD']
            selected_coins = st.session_state['downloaded_coins_list']

            st.markdown(f"<p style='font-size:18px;'><u>Please select all or any of the coins below for prediction and forecasting:</u></p>", unsafe_allow_html=True)
            # st.write("Please select all or any of the coins below for prediction and forecasting.")

            # Create a checkbox to select/deselect all
            new_checkbox_status["prediction_coins_select_all"] = st.checkbox("Select All", value=True, key="prediction_coins_select_all_00")

            # Store the state of individual checkboxes in a dictionary
            prediction_coins_selection = {}

            # Define the number of columns per row (5 in this case)
            num_columns = 4

            # Create rows with columns
            for i in range(0, len(prediction_coins), num_columns):
                cols = st.columns(num_columns)  # Create a row with `num_columns` columns
                for j, coin in enumerate(prediction_coins[i:i + num_columns]):
                    with cols[j]:  # Place each checkbox inside its respective column
                        if new_checkbox_status["prediction_coins_select_all"]:
                            prediction_coins_selection[coin] = st.checkbox(coin, value=True,
                                                                           key=f"prediction_coins{i}_{j}")
                        else:
                            prediction_coins_selection[coin] = st.checkbox(coin, key=f"prediction_coins{i}_{j}")

            selected_coins4 = [coin for coin, selected in prediction_coins_selection.items() if selected]
            if selected_coins4:
                st.write("You selected:")
                st.write(join_with_and(selected_coins4))
            else:
                st.write("No coins selected.")


            st.write("")
            st.markdown(f"<p style='font-size:18px;'><u>Please select all or any of the models below for prediction and forecasting:</u></p>", unsafe_allow_html=True)
            # st.write("Please select all or any of the models below for prediction and forecasting.")

            # Create a checkbox to select/deselect all
            select_all = st.checkbox("Select All Models", value=True, key="model_select_all")

            # Store the state of individual checkboxes in a dictionary
            models_selection = {}

            # Define the number of columns per row (5 in this case)
            num_columns = 4

            # Create rows with columns
            for i in range(0, len(prediction_models), num_columns):
                cols = st.columns(num_columns)  # Create a row with `num_columns` columns
                for j, model in enumerate(prediction_models[i:i + num_columns]):
                    with cols[j]:  # Place each checkbox inside its respective column
                        if select_all:
                            models_selection[model] = st.checkbox(model, value=True,
                                                                           key=f"prediction_models{i}_{j}")
                        else:
                            models_selection[model] = st.checkbox(model, key=f"prediction_models{i}_{j}")

            selected_models = [model for model, selected in models_selection.items() if selected]
            if selected_models:
                st.write("You selected:")
                st.write(join_with_and(selected_models))
            else:
                st.write("No coins selected.")


            st.write("")
            duration = st.number_input("Forecast Duration (days):", min_value=1, max_value=365, value=28)


            st.write("")
            st.markdown(f"<p style='font-size:18px;'><u>Please select all or any of the prediction and forecasting tasks below:</u></p>", unsafe_allow_html=True)
            # st.write("Please select all or any of the prediction and forecasting tasks below.")

            # Create a checkbox to select/deselect all
            new_checkbox_status["prediction_select_all"] = st.checkbox("Select All", key="predict_select_all")

            # Store the state of individual checkboxes in a dictionary
            prediction_checkbox_states = {}

            # Define the number of columns per row (5 in this case)
            num_columns = 4

            # Create rows with columns
            for i in range(0, len(prediction_tasks), num_columns):
                cols = st.columns(num_columns)  # Create a row with `num_columns` columns
                for j, task in enumerate(prediction_tasks[i:i + num_columns]):
                    with cols[j]:  # Place each checkbox inside its respective column
                        if new_checkbox_status["prediction_select_all"]:
                            prediction_checkbox_states[task] = st.checkbox(task, value=True,
                                                                  key=f"prediction_tasks{i}_{j}")
                        else:
                            prediction_checkbox_states[task] = st.checkbox(task, key=f"prediction_tasks{i}_{j}")

            selected_prediction_tasks = [task for task, selected in prediction_checkbox_states.items() if selected]
            if selected_prediction_tasks:
                st.write("You selected:")
                st.write(join_with_and(selected_prediction_tasks))
            else:
                st.write("No tasks selected.")


            # selected_prediction_tasks = [task for task, selected in st.session_state.old_checkbox_status.items() if selected and task in prediction_tasks]
            if selected_prediction_tasks:
                missing_data = [x for x in selected_coins4 if x not in selected_coins]

                if missing_data:
                    st.error(f'Error:---Please download data for {missing_data} to proceed')
                elif selected_coins4 == []:
                    st.error(f'Error:---Please select at least 1 coin for "Prediction and Forecasting".')
                elif selected_models == []:
                    st.error(f'Error:---Please select at least 1 model for "Prediction and Forecasting".')
                else:
                    data = st.session_state.data["Cleaned Data"]["All Coins"]
                    # st.write(data.shape)
                    pred_data = data[(data['Crypto'].isin(sorted(selected_coins4)))]
                    # Reset index to flatten the DataFrame
                    pred_data.reset_index(inplace=True, drop=True)
                    st.write("---")
                    text = f"Prediction and Forecasting for Selected Cryptocurrencies:"

                    st.write(f"## {text}")



                    call_prediction_model = {}
                    prediction_models = {
                        'Prophet': predict_with_prophet,
                        'ARIMA': predict_with_arima,
                        'LSTM': predict_with_lstm,
                        'SVM': predict_with_svm
                    }

                    call_forecast_model = {}
                    forecast_models = {
                        'Prophet': forecast_with_prophet,
                        'ARIMA': forecast_with_arima,
                        'LSTM': forecast_with_lstm,
                        'SVM': forecast_with_svm
                    }

                    # st.write(data.shape)
                    if "Show Heatmap" in selected_prediction_tasks:
                        combined_data_heatmap(selected_coins, data, selected_coins4)


                    # Perform prediction if selected
                    if "Predict" in selected_prediction_tasks:

                        # data = st.session_state.data[data_keys[0]]
                        prophet_prediction=arima_prediction=lstm_prediction=svm_prediction=xgb_prediction={}

                        for model_type in selected_models:
                            # st.write(f"## {model_type} Prediction Chart for {coin}")
                            st.write(f"## {model_type} Prediction Chart for")
                            if model_type == 'Prophet':
                                call_prediction_model[f"{model_type}"] = predict_with_prophet(pred_data, show_plot=True)
                            elif model_type == 'ARIMA':
                                call_prediction_model[f"{model_type}"] = predict_with_arima(pred_data, show_plot=True)
                            elif model_type == 'LSTM':
                                call_prediction_model[f"{model_type}"] = predict_with_lstm(pred_data, show_plot=True)
                            elif model_type == 'SVM':
                                call_prediction_model[f"{model_type}"] = predict_with_svm(pred_data, show_plot=True)
                            else:
                                st.error("Invalid model type selected")
                                st.stop()

                    # Perform prediction if selected
                    if "Forecast" in selected_prediction_tasks:

                        # data = st.session_state.data[data_keys[0]]

                        for model_type in selected_models:
                            # st.write(f"## {model_type} Prediction Chart for {coin}")
                            st.write(f"## {model_type} Forecast Chart for")
                            if model_type == 'Prophet':
                                call_forecast_model[f"{model_type}"] = forecast_with_prophet(pred_data, duration, show_plot=True)
                            elif model_type == 'ARIMA':
                                call_forecast_model[f"{model_type}"] = forecast_with_arima(pred_data, duration, show_plot=True)
                            elif model_type == 'LSTM':
                                call_forecast_model[f"{model_type}"] = forecast_with_lstm(pred_data, duration, show_plot=True)
                            elif model_type == 'SVM':
                                call_forecast_model[f"{model_type}"] = forecast_with_svm(pred_data, duration, show_plot=True)
                            else:
                                st.error("Invalid model type selected")
                                st.stop()
                            # plot_predictions(data, prediction)

            else:
                st.info(f'Info:---Please select at least 1 of "Prediction and Forcasting" tasks to proceed.')



    except Exception as e:
        st.error(f"Error: {e}")
        st.error(traceback.format_exc())

