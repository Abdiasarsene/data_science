import streamlit as st
import pandas as pd

# Charger le fichier css
with open('style.css') as css_file:
    css = css_file.read()

# Afficher le fichier css
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Charger le fichier html
with open('index.html') as html_file:
    html_content = html_file.read()

# Afficher le fichier html
st.markdown(html_content, unsafe_allow_html=True)

# Ajouter une navigation latérale
st.sidebar.title('Navigation')
option = st.sidebar.radio('Go to', ['estateIQ','DataViz', 'Prédictions', 'Résultats'])

# Ajouter écrivez votre texte
topic = st.text_input('Your topic',key='topic_input')
describe = st.text_area('More descriptions', height=200)

# Importer votre base de doonnée
uploader_file = st.file_uploader('Import your database', type=['csv', 'xlsx','dta','sav'])

# Vérification du fichier uploader
if uploader_file is not None:
    if uploader_file.name.endswith('csv'):
        try:
            database = pd.read_csv(uploader_file, encoding='ISO-8859-1')
        except UnicodeDecodeError:
            database = pd.read_csv(uploader_file, encoding='cp1252')

    elif uploader_file.name.endswith('xlsx') or uploader_file.name.endswith('xls') :
        database =pd.read_excel(uploader_file)
    elif uploader_file.name.endswith('sav'):
        database =pd.read_spss(uploader_file)
    elif uploader_file.name.endswith('dta'):
        database = pd.read_stata(uploader_file)
    # Affichage de la base de données
    st.success('Your import are successful')
else :
    st.warning("Please upload a .csv, .xlsx, .xls, .dta, or .sav file.")

# Créer une redirection sur la page suivante
if st.button('Proceed'):
    if not topic:
        st.warning('Please, you need to write your topic')
    elif not describe:
        st.warning('Please, you need to write your description')
    elif uploader_file is None:
        st.warning('Important to import your database')
    else:
        st.success('All fields are correctly filled')

# Redirection vers la page 'DataViz' si cette option est sélectionnée et que la base de données est disponible
if option == 'DataViz':
    if 'database' in locals():
        from app_ml import dataviz
        dataviz.main(database)
    else:
        st.warning("Please import your database to visualize it.")
