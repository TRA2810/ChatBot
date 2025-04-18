import nltk
import streamlit as st 
# télécharger les ressources nécessaires pour découper un texte en phrase et en mots (tokénisation)
nltk.download('punkt_tab')  # nécessaire pour découper un texte en phrase et en mots (tokenisation)
nltk.download('averaged_perceptron_tagger')  # nécessaire pour connaitre la nature des mots (nom, verbe, adjectif)
nltk.download('stopwords')  # liste de mots courants inutile pour l'analyse (le, la, de, et...), à supprimer du texte
nltk.download('wordnet')  # dictionnaire lexical pour faire de la lemmatisation
nltk.download('omw-1.4')  # nécessaire pour que WordNet puisse fonctionner correctement

import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize  # sent_tokenize : utilisé pour découper le texte en phrase
from nltk.stem import WordNetLemmatizer  # WordNetLemmatizer permet de faire de la lemmatisation

# Chargement du texte (en français ici)
with open('pub_entreprise.txt', 'r', encoding='utf-8') as f:
    data = f.read().replace('\n', '')

# Découpage du texte en phrases

sentences = sent_tokenize(data)

# Liste des mots inutiles (stopwords)
stop_words = set(stopwords.words('french'))

# Fonction de prétraitement : tokenisation, nettoyage et lemmatisation
def preprocess(sentence):
    words = word_tokenize(sentence, language='french')  # Découpe la phrase en mots
    words = [
        word.lower()  # Mettre tous les mots en minuscule
        for word in words  # Parcourir la liste des mots
        if word.lower() not in stop_words and word not in string.punctuation
    ]
    
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatisation des mots

    return words

# Traitement du corpus
corpus = [preprocess(sentence) for sentence in sentences]  # Corpus prétraité : une collection de textes ou de phrases

# Fonction de recherche de la phrase la plus pertinente
def get_most_relevant_sentence(query):  # Fonction pour obtenir la phrase la plus pertinente
    query = preprocess(query)  # Prétraiter la requête de l'utilisateur
    max_similarity = -1  # Initialisation de la similarité maximale
    most_relevant_sentence = ""  # Phrase la plus pertinente

    # Comparaison de la requête avec chaque phrase du corpus
    for i, sentence in enumerate(corpus):
        # Calcul de la similarité entre la requête et chaque phrase
        intersection = len(set(query).intersection(set(sentence)))
        union = len(set(query).union(set(sentence)))
        similarity = intersection / float(union)  # Similarité de Jaccard (nombre de mots communs / nombre total de mots uniques)

        # Si la similarité est supérieure à celle trouvée précédemment
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_sentence = sentences[i]  # Mise à jour de la phrase la plus pertinente

    return most_relevant_sentence


# Interface streamlit 
st.title('chatbot eclat brillant')
question=st.text_input('posez des questions sur nos produits: ')
if question:
    response=get_most_relevant_sentence(question)
    st.write('Reponse:',response)
