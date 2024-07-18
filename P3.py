import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Part III: Visualization
def part_3_visualization():
    st.header("Part III: Visualization")
    data = st.session_state.get('data')

    if data is not None:
        numerical_columns = data.select_dtypes(include=['number']).columns.tolist()

        # Histograms
        st.subheader("Histograms of Numerical Features")
        if len(numerical_columns) > 1:
            fig, axes = plt.subplots(nrows=1, ncols=len(numerical_columns), figsize=(15, 5))
            for i, col in enumerate(numerical_columns):
                sns.histplot(data[col], kde=True, ax=axes[i])
            st.pyplot(fig)
        else:
            st.write("Histograms cannot be plotted as there is only one numerical column.")

        # Box Plots
        st.subheader("Box plots of Numerical Features")
        if len(numerical_columns) > 1:
            fig, axes = plt.subplots(nrows=1, ncols=len(numerical_columns), figsize=(15, 5))
            for i, col in enumerate(numerical_columns):
                sns.boxplot(x=data[col], ax=axes[i])
            st.pyplot(fig)
        else:
            st.write("Box plots cannot be plotted as there is only one numerical column.")

        # Scatter Plots
        st.subheader("Scatter plots of Numerical Features")
        if len(numerical_columns) > 1:
            fig, axes = plt.subplots(nrows=len(numerical_columns), ncols=len(numerical_columns), figsize=(15, 15))
            for i, col1 in enumerate(numerical_columns):
                for j, col2 in enumerate(numerical_columns):
                    if i != j:
                        sns.scatterplot(x=data[col1], y=data[col2], ax=axes[i,j])
                    else:
                        axes[i,j].text(0.5, 0.5, col1, fontsize=12, ha='center')
                        axes[i,j].axis('off')
            st.pyplot(fig)
        else:
            st.write("Scatter plots cannot be plotted as there is only one numerical column.")
            
        # Correlation Matrix
        st.subheader("Correlation Matrix")
        if len(numerical_columns) > 1:
            corr = pd.DataFrame(data, columns=numerical_columns).corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:    
            st.write("Correlation matrix cannot be computed as there is only one numerical column.")
    else:
        st.error("No data available. Please upload data in Part I.")