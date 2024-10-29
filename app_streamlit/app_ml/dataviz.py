import streamlit as st
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns 

def main(database):
    # Display HTML content
    with open('dataviz.html') as dataviz_html:
        html_content = dataviz_html.read()
    st.markdown(html_content, unsafe_allow_html=True)

    # Display CSS content
    with open('style.css') as css_file:
        css = css_file.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

    # Display the last few rows of the database
    st.markdown("""
        <h2>Print of database</h2>
        <br>
    """, unsafe_allow_html=True)
    st.write(database.tail())

    # Display title for missing values section
    st.markdown("""
        <h2>Missing Values</h2>
        <br>
""", unsafe_allow_html=True)

    # Plot the missing values graphic
    missing_data = database.isna().sum().sum()
    if missing_data == 0 :
        st.success("Your database has no missing data, great job")
        fig, ax = plt.subplots( figsize=(18,5))
        msno.bar(database, ax= ax, color='orange')
        # Affichage du graphique
        st.pyplot(fig)
    else :
        st.warning("Your database contains missing data")
        fillna_data=database.fillna(0)
        st.write(fillna_data)
    
    # Détections les valeurs manquantes
    Q1 = database.quantile(0.25) 
    Q3 = database.quantile(0.25)
    
    IQR = Q3 - Q1
    
    outliers = ((database< (Q1-1.5*IQR) |(database> (Q3 +1.5*IQR))))
    
    valeur_aberrantes = database[outliers.any(axis=1)] 
    
    if not valeur_aberrantes.empty:
        st.warning('Datasets contains outliers')
        st.write(valeur_aberrantes.head())
        
        database_cleaned = database.drop(valeur_aberrantes.index)
        st.write(database_cleaned.tail())
        st.info('The database is cleaned now, download the new dtatbase')
    else:
        st.success('Dataset is clean, no outliers detected')

    # Réaliser la statistique descriptive
    st.markdown("""
        <h2>Descriptive Statistic</h2>
        <br>
    """, unsafe_allow_html=True)
    statistic = database.describe()
    st.write(statistic)
    
    plot_reg_and_pair(database_cleaned)

def plot_reg_and_pair(database):
    # Display regplot
    st.markdown("""
        <h2> DataViz : Regplot</h2>
        <br>
    """, unsafe_allow_html=True)
    st.write("The regression plot below shows the linear relationship between two chosen variables.")

    # Select columns for regplot
    numeric_columns = database.select_dtypes(include='number').columns.tolist()
    x_col = st.selectbox("Choose X axis for regplot", numeric_columns, index=0)
    y_col = st.selectbox("Choose Y axis for regplot", numeric_columns, index=1)

    fig, ax = plt.subplots(figsize=(18,5))
    sns.regplot(x=x_col, y=y_col, data=database, ax=ax, color='purple')
    st.pyplot(fig)

    # Display pairplot
    st.markdown("""
        <h2> DataViz : Pairplot</h2>
        <br>
    """, unsafe_allow_html=True)    
    st.write("The pair plot shows scatter plots and histograms for each numeric variable pair in the dataset.")

    # Generate pairplot for all numeric columns
    pair_fig = sns.pairplot(database[numeric_columns])
    st.pyplot(pair_fig)

