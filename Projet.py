from P1 import part_1_initial_data_exploration
from P2 import part_2_data_preprocessing_and_cleaning
from P3 import part_3_visualization
from P4 import part_4_clustering_or_prediction
import streamlit as st


def main():
    # Set up the Streamlit app
    st.title("Data Analysis and Clustering Web Application")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    part = st.sidebar.radio("Go to", ["Part I: Initial Data Exploration", 
                                    "Part II: Data Pre-processing and Cleaning", 
                                    "Part III: Visualization", 
                                    "Part IV: Clustering or Prediction"])

    # Main logic to call appropriate function based on sidebar selection
    if part == "Part I: Initial Data Exploration":
        part_1_initial_data_exploration()
    elif part == "Part II: Data Pre-processing and Cleaning":
        part_2_data_preprocessing_and_cleaning()
    elif part == "Part III: Visualization":
        part_3_visualization()
    elif part == "Part IV: Clustering or Prediction":
        part_4_clustering_or_prediction()

main()
