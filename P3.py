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
            for col in numerical_columns:
                fig, ax = plt.subplots()
                sns.histplot(data[col], ax=ax, kde=True)
                st.pyplot(fig)
        else:
            st.write("Histograms cannot be plotted as there is only one numerical column.")

        # Box Plots
        st.subheader("Box plots of Numerical Features")
        if len(numerical_columns) > 1:
            for col in numerical_columns:
                fig, ax = plt.subplots()
                sns.boxplot(x=data[col], ax=ax)
                st.pyplot(fig)
        else:
            st.write("Box plots cannot be plotted as there is only one numerical column.")

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