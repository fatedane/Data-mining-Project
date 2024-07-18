import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Fonction pour ajouter les styles CSS personnalisés
def add_custom_css():
    st.markdown(
        """
        <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 30px;
        }
        .header {
            font-size: 24px;
            font-weight: bold;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        .stButton > button:first-child {
            background-color: #4CAF50;
            color: white;
            font-size: 14px;
        }
        .stSlider > .stSlider-container > div {
            background-color: #4CAF50;
        }
        .stSelectbox > div:first-child {
            background-color: #f0f0f0;
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 10px;
        }
        .stTable {
            background-color: #f0f0f0;
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 10px;
        }
        .stBlockquote {
            background-color: #f9f9f9;
            border-left: 10px solid #4CAF50;
            padding: 10px;
            margin-top: 20px;
        }
        .stMarkdown {
            color: #333;
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Set up the Streamlit app
add_custom_css()
st.title("Data Analysis and Clustering Web Application")

# Part I: Initial Data Exploration
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

    # Part II: Data Pre-processing and Cleaning
    st.header("Part II: Data Pre-processing and Cleaning")

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

    # Histograms
    st.subheader("Histograms of Numerical Features")
    for col in numerical_columns:
        fig, ax = plt.subplots()
        sns.histplot(data[col], ax=ax, kde=True)
        st.pyplot(fig)

    # Box Plots
    st.subheader("Box plots of Numerical Features")
    for col in numerical_columns:
        fig, ax = plt.subplots()
        sns.boxplot(x=data[col], ax=ax)
        st.pyplot(fig)

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    corr = pd.DataFrame(data, columns=numerical_columns).corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Part IV: Clustering or Prediction
    st.header("Part IV: Clustering or Prediction")
    task = st.selectbox("Choose a task:", ["Clustering", "Prediction"])

    if task == "Clustering":
        clustering_algo = st.selectbox("Choose a clustering algorithm:", ["K-Means", "DBSCAN"])
        
        if clustering_algo == "K-Means":
            n_clusters = st.slider("Choose number of clusters:", 2, 10, 3)
            kmeans = KMeans(n_clusters=n_clusters)
            clusters = kmeans.fit_predict(data[numerical_columns])
            data['Cluster'] = clusters
            
            st.write("Cluster centers:")
            st.write(kmeans.cluster_centers_)
        
        elif clustering_algo == "DBSCAN":
            eps = st.slider("Choose eps parameter:", 0.1, 10.0, 0.5)
            min_samples = st.slider("Choose min_samples parameter:", 1, 20, 5)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(data[numerical_columns])
            data['Cluster'] = clusters
        
        # Visualization of Clusters
        st.subheader("2D scatter plot of clusters")
        pca = PCA(2)
        pca_data = pca.fit_transform(data[numerical_columns])
        fig, ax = plt.subplots()
        scatter = ax.scatter(pca_data[:,0], pca_data[:,1], c=data['Cluster'])
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        st.pyplot(fig)

        st.subheader("3D scatter plot of clusters")
        pca = PCA(3)
        pca_data = pca.fit_transform(data[numerical_columns])
        pca_df = pd.DataFrame(data=pca_data, columns=['PCA1', 'PCA2', 'PCA3'])
        pca_df['Cluster'] = data['Cluster']

        # Create a 3D scatter plot using Plotly
        fig = px.scatter_3d(
            pca_df, x='PCA1', y='PCA2', z='PCA3',
            color='Cluster', title='3D scatter plot of clusters'
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)
        
        # Cluster Statistics
        st.subheader("Cluster statistics")
        st.write(data['Cluster'].value_counts())

    elif task == "Prediction":
        prediction_type = st.selectbox("Choose a prediction type:", ["Regression", "Classification"])
        if prediction_type == "Regression":
            data = data.drop(columns=categorical_columns)
            target_column = st.selectbox("Select the target column for prediction:", data.columns.tolist())
            X = data.drop(columns=[target_column])
            y = data[target_column]

            regression_algo = st.selectbox("Choose a regression algorithm:", ["Linear Regression", "Ridge Regression", "Lasso Regression"])

            if regression_algo == "Linear Regression":
                model = LinearRegression()
            elif regression_algo == "Ridge Regression":
                alpha = st.slider("Choose the alpha parameter for Ridge Regression:", 0.01, 10.0, 1.0)
                model = Ridge(alpha=alpha)
            elif regression_algo == "Lasso Regression":
                alpha = st.slider("Choose the alpha parameter for Lasso Regression:", 0.01, 10.0, 1.0)
                model = Lasso(alpha=alpha)

            test_size = st.slider("Choose test size (fraction of data to be used as test set):", 0.1, 0.5, 0.2)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
            st.write("Regression Coefficients:", model.coef_)
            if regression_algo != "Linear Regression":
                st.write("Intercept:", model.intercept_)

        elif prediction_type == "Classification":
            data_s = data.drop(columns=numerical_columns)
            target_column = st.selectbox("Select the target column for prediction:", data_s.columns.tolist())
            X = data.drop(columns=[target_column])
            y = data[target_column]

            classification_algo = st.selectbox("Choose a classification algorithm:", ["K-Nearest Neighbors"])
            
            if classification_algo == "K-Nearest Neighbors":
                n_neighbors = st.slider("Choose number of neighbors:", 1, 20, 5)
                test_size = st.slider("Choose test size (fraction of data to be used as test set):", 0.1, 0.5, 0.2)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                model = KNeighborsClassifier(n_neighbors=n_neighbors)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                st.write("Accuracy Score:", accuracy_score(y_test, y_pred))
                st.write("Confusion Matrix:")
                st.write(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))

# Run the Streamlit app
# To run the app, save this script and execute `streamlit run script_name.py` in the terminal
