import streamlit as st
import pandas as pd

# Part I: Initial Data Exploration
def part_1_initial_data_exploration():
    st.header("Part I: Initial Data Exploration")

    # Data Loading
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        delimiter = st.text_input("Enter the delimiter (e.g., ',' for CSV):", value=',')
        header_option = st.radio("Does your file have a header row?", ('Yes', 'No'))
        
        if header_option == 'Yes':
            header = int(st.text_input("Enter the header row index (e.g., 0 for the first row):", value=0))
            data = pd.read_csv(uploaded_file, delimiter=delimiter, encoding='utf-8', header=header)
        else:
            data = pd.read_csv(uploaded_file, delimiter=delimiter, header=None)
            st.write("Data loaded without headers. Please enter column names.")
            columns = st.text_input("Enter column names separated by commas (e.g., col1,col2,col3):")
            if columns:
                column_list = columns.split(',')
                if len(column_list) == data.shape[1]:
                    data.columns = column_list
                else:
                    st.error(f"Error: You have provided {len(column_list)} column names, but the dataset has {data.shape[1]} columns. Please provide the correct number of column names.")

        # Display column names for selection
        st.write("Current columns in the dataset:")
        st.write(data.columns.tolist())
        
        # Allow user to select columns to drop
        columns_to_drop = st.multiselect("Select columns to drop:", data.columns.tolist())
        if columns_to_drop:
            data = data.drop(columns=columns_to_drop)
            st.write(f"Columns {columns_to_drop} have been dropped.")

        st.write("First few rows of the data:")
        st.write(data.head())
        st.write("Last few rows of the data:")
        st.write(data.tail())

        # Data Description
        st.write("Data Description:")
        st.write(data.describe(include='all'))
        st.write("Number of missing values per column:")
        st.write(data.isnull().sum())
        
        # Store data in session state for later use
        st.session_state['data'] = data