import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler,LabelEncoder

# Part II: Data Pre-processing and Cleaning
def part_2_data_preprocessing_and_cleaning():
    st.header("Part II: Data Pre-processing and Cleaning")
    data = st.session_state.get('data')

    if data is not None:
        # Separate Numerical and Non-Numerical Data
        numerical_columns = data.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        st.write("Numerical columns detected:", numerical_columns)
        st.write("Categorical columns detected:", categorical_columns)

        # Handling Non-Numerical Data
        st.subheader("Handling Non-Numerical Data")

        encoding_method = st.selectbox(
            "Choose a method to encode categorical data:",
            ["None", "One-Hot Encoding", "Label Encoding"]
        )

        if encoding_method == "One-Hot Encoding":
            data = pd.get_dummies(data, columns=categorical_columns)
        elif encoding_method == "Label Encoding":
            for col in categorical_columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])

        st.write("Data after encoding:")
        st.write(data.head())

        # Handling Missing Values
        st.subheader("Handling Missing Values")

        # Separate handling for numerical and categorical data
        missing_value_method_num = st.selectbox(
            "Choose a method to handle missing values in numerical data:",
            ["Delete rows/columns", "Replace with mean", "Replace with median", "Replace with mode", "KNN Imputation"]
        )

        missing_value_method_cat = st.selectbox(
            "Choose a method to handle missing values in categorical data:",
            ["Delete rows/columns", "Replace with most frequent", "Replace with constant"]
        )

        # Handle missing values in numerical data
        if missing_value_method_num == "Delete rows/columns":
            data = data.dropna(subset=numerical_columns)
        elif missing_value_method_num == "Replace with mean":
            imputer = SimpleImputer(strategy='mean')
            data[numerical_columns] = imputer.fit_transform(data[numerical_columns])
        elif missing_value_method_num == "Replace with median":
            imputer = SimpleImputer(strategy='median')
            data[numerical_columns] = imputer.fit_transform(data[numerical_columns])
        elif missing_value_method_num == "Replace with mode":
            imputer = SimpleImputer(strategy='most_frequent')
            data[numerical_columns] = imputer.fit_transform(data[numerical_columns])
        elif missing_value_method_num == "KNN Imputation":
            imputer = KNNImputer()
            data[numerical_columns] = imputer.fit_transform(data[numerical_columns])

        # Handle missing values in categorical data
        if missing_value_method_cat == "Delete rows/columns":
            data = data.dropna(subset=categorical_columns)
        elif missing_value_method_cat == "Replace with most frequent":
            imputer = SimpleImputer(strategy='most_frequent')
            data[categorical_columns] = imputer.fit_transform(data[categorical_columns])
        elif missing_value_method_cat == "Replace with constant":
            constant_value = st.text_input("Enter the constant value for replacement:")
            imputer = SimpleImputer(strategy='constant', fill_value=constant_value)
            data[categorical_columns] = imputer.fit_transform(data[categorical_columns])

        st.write("Data after handling missing values:")
        st.write(data.head())

        # Data Normalization
        st.subheader("Data Normalization")

        normalization_method = st.selectbox(
            "Choose a normalization method:",
            ["None", "Min-Max Normalization", "Z-score Standardization"]
        )

        if normalization_method == "Min-Max Normalization":
            scaler = MinMaxScaler()
            data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        elif normalization_method == "Z-score Standardization":
            scaler = StandardScaler()
            data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

        st.write("Data after normalization:")
        st.write(data.head())

        # Store data in session state for later use
        st.session_state['data'] = data
    else:
        st.error("No data available. Please upload data in Part I.")