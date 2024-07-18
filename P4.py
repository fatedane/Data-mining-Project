import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Part IV: Clustering or Prediction
def part_4_clustering_or_prediction():
    st.header("Part IV: Clustering or Prediction")
    data = st.session_state.get('data')

    if data is not None:
        numerical_columns = data.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        
        if len(numerical_columns) < 2:
            st.error("Error: Clustering and Prediction tasks require at least 2 numerical columns. Please provide a dataset with more numerical columns.")
        else:
            task = st.selectbox("Choose a task:", ["Clustering", "Prediction"])

            if task == "Clustering":
                clustering_algo = st.selectbox("Choose a clustering algorithm:", ["K-Means", "DBSCAN"])
                
                if clustering_algo == "K-Means":
                    n_clusters = st.slider("Choose number of clusters:", 2, 10, 3)
                    kmeans = KMeans(n_clusters=n_clusters)
                    clusters = kmeans.fit_predict(data[numerical_columns])

                    # Create a copy of data for visualization without modifying original data
                    data_vis = data.copy()
                    data_vis['Cluster'] = clusters

                    st.write("Cluster centers:")
                    st.write(kmeans.cluster_centers_)
                
                elif clustering_algo == "DBSCAN":
                    eps = st.slider("Choose eps parameter:", 0.1, 10.0, 0.5)
                    min_samples = st.slider("Choose min_samples parameter:", 1, 20, 5)
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    clusters = dbscan.fit_predict(data[numerical_columns])

                    # Create a copy of data for visualization without modifying original data
                    data_vis = data.copy()
                    data_vis['Cluster'] = clusters
                
                # Visualization of Clusters
                if clustering_algo in ["K-Means", "DBSCAN"]:
                    st.subheader("2D scatter plot of clusters")
                    pca = PCA(2)
                    pca_data = pca.fit_transform(data[numerical_columns])
                    pca_vis_data = pca.transform(data_vis[numerical_columns])
                    fig, ax = plt.subplots()
                    scatter = ax.scatter(pca_vis_data[:,0], pca_vis_data[:,1], c=data_vis['Cluster'])
                    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
                    ax.add_artist(legend1)
                    st.pyplot(fig)

                    st.subheader("3D scatter plot of clusters")
                    pca = PCA(3)
                    pca_data = pca.fit_transform(data[numerical_columns])
                    pca_vis_data = pca.transform(data_vis[numerical_columns])
                    pca_df = pd.DataFrame(data=pca_vis_data, columns=['PCA1', 'PCA2', 'PCA3'])
                    pca_df['Cluster'] = data_vis['Cluster']

                    fig = px.scatter_3d(
                        pca_df, x='PCA1', y='PCA2', z='PCA3',
                        color='Cluster', title='3D scatter plot of clusters'
                    )
                    st.plotly_chart(fig)
                    
                    st.subheader("Cluster statistics")
                    st.write(data_vis['Cluster'].value_counts())

            elif task == "Prediction":
                prediction_type = st.selectbox("Choose a prediction type:", ["Regression", "Classification"])
                if prediction_type == "Regression":
                    data_reg = data.drop(columns=categorical_columns)
                    target_column = st.selectbox("Select the target column for prediction:", data_reg.columns.tolist())
                    X = data_reg.drop(columns=[target_column])
                    y = data_reg[target_column]

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
                    data_vis2= data.copy()
                    data_clf = data.drop(columns=numerical_columns)
                    target_column = st.selectbox("Select the target column for prediction:", data_clf.columns.tolist())
                    X = data_vis2.drop(columns=[target_column])
                    y = data_vis2[target_column]

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
    else:
        st.error("No data available. Please upload data in Part I.")
