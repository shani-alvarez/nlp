# Text normalization class

import pandas as pd
import demoji
import re

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import spacy

class TextNormalization:
    
    def __init__(self, directory):
        self.directory = directory # Working directory
        self.sp_en = spacy.load('en_core_web_trf') #english
        self.sp_sp = spacy.load('es_dep_news_trf') #spanish
    
    def tokenizeText(self, text):
        # Tokenización por palabras individuales
        token_list = word_tokenize(text)
        # Eliminación de tokens con una longitud < 2
        token_list = [token for token in token_list if len(token) > 1]
        return token_list

    def tokenizeTweets(self, data):
        # Aplica la función de limpieza y tokenización a cada texto
        data["tokenized_text"] = data["text"].apply(lambda x: self.tokenizeText(x))
        return data
    
    # https://www.cienciadedatos.net/documentos/py25-text-mining-python.html
    # Proyecto demoji: https://pypi.org/project/demoji/#description
    def cleanText(self, list_tokens):
        new_list_tokens = []
        for text in list_tokens:
            #Elimina los emojis
            new_text = demoji.replace(text, '')
            # Se convierte todo el texto a minúsculas
            new_text = text.lower()
            # Eliminación de páginas web (palabras que empiezan por "http")
            new_text = re.sub('http\S+', '', new_text)
            # Eliminación de páginas web (palabras que empiezan por "//")
            new_text = re.sub('//\S+', '', new_text)
            # Eliminación de signos de puntuación
            regex = '[\\!\\¡\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\¿\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]'
            new_text = re.sub(regex , '', new_text)
            # Eliminación de números
            new_text = re.sub("\d+", '', new_text)
            # Eliminación de espacios en blanco múltiples
            new_text = re.sub("\\s+", '', new_text)
            # Add new text to the new token list
            if new_text != '':
                new_list_tokens.append(new_text)
            else: 
                pass
        return(new_list_tokens)

    def cleanTweets(self, data):
        data["clean_tokenized_text"] = data["tokenized_text"].apply(lambda x: self.cleanText(x))
        return data
    
    def removeStopwords(self, token_list):
        # List of stopwords in english and spanish
        stop_words_en = list(stopwords.words('english'))
        stop_words_esp = list(stopwords.words('spanish'))
        stop_words = stop_words_en + stop_words_esp
        stop_words.extend(("amp", "xa", "xe", "si"))  # Se añade la stopword: amp, ax, ex, ""
        filtered_token_list = [word for word in token_list if word not in stop_words]
        return filtered_token_list

    def removeStopwordsInTokenList(self, data):
        data["filtered_tokenized_text"] = data["clean_tokenized_text"].apply(lambda x: self.removeStopwords(x))
        return data
    
    def textLemmatization(self, list_words):
        lemma_list_words = []
        for i in range(0, len(list_words)):
            words = self.sp_sp(list_words[i])
            for word2 in words:
                lemma_list_words.append(word2.lemma_)
        return lemma_list_words
        
    def tweetLemmatization(self, data):
        # Aplica la función de limpieza y tokenización a cada texto
        data["lemmatized_normal_text"] = data["filtered_tokenized_text"].apply(lambda x: self.textLemmatization(x))
        return data