# Text analysis class

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import LatentDirichletAllocation

class TextAnalysis:
    
    def __init__(self, directory):
        self.directory = directory # Working directory
        
    def stopwords(self):
        # Obtiene el listado de stopwords en inglés y español
        stop_words_en = list(stopwords.words('english'))
        stop_words_esp = list(stopwords.words('spanish'))
        stop_words = stop_words_en + stop_words_esp
        # Se añade la stopword: amp, ax, ex
        stop_words.extend(("amp", "xa", "xe", "si"))
        return stop_words
    
    def unigramsTweets(self, data):
        df = data.copy()
        df.loc[:, ["user_id", "screen_name", "followers_count", "retweet_count", "favourites_count", "friends_count", "lemmatized_normal_text"]]
        print("Número de NaNs en la columna 'lemmatized_normal_text': " + str(df.lemmatized_normal_text.isna().sum()) + "\n")

        # Unnest de la columna tokenized_description
        df_analysis_tidy = df.explode(column='lemmatized_normal_text')
        df_analysis_tidy = df_analysis_tidy.rename(columns={'lemmatized_normal_text':'token_text'})
        #Dataframe con tokens de los tweets
        print("Shape del tidy dataframe : " + str(df_analysis_tidy.shape) + "\n")
        token_texto_conteo = df_analysis_tidy.token_text.value_counts().rename_axis('token_text').reset_index(name='count')
        # Convertir frecuencias a diccionarios
        dict_texto = dict(zip(token_texto_conteo['token_text'].tolist(), token_texto_conteo['count'].tolist()))

        #Gráfica de nube de palabras
        print('Nube de palabras con unigramas:', "\n")
        wc_texto = WordCloud(width=800, height=400, max_words=100, background_color="white").generate_from_frequencies(dict_texto)
        plt.figure(figsize=(10, 10))
        plt.imshow(wc_texto, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        
        return df_analysis_tidy, token_texto_conteo
        
    def ngramsTweets(self, data):
        df = data.copy()
        df["lemmatized_normal_tweet"] = df["lemmatized_normal_text"].apply(lambda x: ' '.join(x))
        data_bigram = df.loc[:, ["user_id", "screen_name", "followers_count", "retweet_count", "favourites_count", "friends_count", "lemmatized_normal_tweet"]]

        # Obtención de listado de stopwords en inglés y español
        stop_words = self.stopwords()

        # Elimina stopwords de la columna clean_text, genera los bigramas/trigramas y obtiene su frecuencia
        from sklearn.feature_extraction.text import CountVectorizer
        c_vec = CountVectorizer(stop_words=stop_words, ngram_range=(2,3))
        # Matriz de ngramas
        ngrams = c_vec.fit_transform(data_bigram['lemmatized_normal_tweet'])
        # Frecuencia de ngramas
        count_values = ngrams.toarray().sum(axis=0)
        # Lista de ngramas
        vocab = c_vec.vocabulary_
        conteo_ngramas = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)).rename(columns={0: 'frecuencia', 1:'bigrama/trigrama'})
        # Convertir frecuencias a diccionarios
        dict_texto_ngramas = dict(zip(conteo_ngramas['bigrama/trigrama'].tolist(), conteo_ngramas['frecuencia'].tolist()))

        # Gráfica de nube de palabras
        print('Nube de palabras con bigramas/trigramas:', "\n")
        wc_texto_ngramas = WordCloud(width=800, height=400, max_words=100, background_color="white").generate_from_frequencies(dict_texto_ngramas)
        plt.figure(figsize=(20, 20))
        plt.imshow(wc_texto_ngramas, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        
        return data_bigram, conteo_ngramas
        
    def print_top_words(self, model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            message += ", ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)
        print()

    def themeModels(self, data, n_components_in, n_top_words_in):
        # Source: https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
        print("NMF model...............................", "\n")
        df = data.copy()
        df["lemmatized_normal_tweet"] = df["lemmatized_normal_text"].apply(lambda x: ' '.join(x))
        df = df.loc[:, ["user_id", "screen_name", "followers_count", "retweet_count", "favourites_count", "friends_count", "lemmatized_normal_tweet"]]
        stop_words = self.stopwords() # Listado de stopwords en inglés y español
        
        tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(2,3))
        nmf = NMF(n_components=n_components_in)
        pipe_nmf = make_pipeline(tfidf_vectorizer, nmf)
        pipe_nmf.fit(df['lemmatized_normal_tweet'])
        self.print_top_words(nmf, tfidf_vectorizer.get_feature_names(), n_top_words=n_top_words_in)
        print()

        print("LDA model...............................", "\n")
        tfidf_vectorizer_2 = TfidfVectorizer(stop_words=stop_words, ngram_range=(2,3))
        lda = LatentDirichletAllocation(n_components=n_components_in)
        pipe_lda = make_pipeline(tfidf_vectorizer_2, lda)
        pipe_lda.fit(df['lemmatized_normal_tweet'])
        self.print_top_words(lda, tfidf_vectorizer_2.get_feature_names(), n_top_words=n_top_words_in)
    
    def tweetsAnalysis(self, data, n_components_in, n_top_words_in):
        self.unigramsTweets(data)
        self.ngramsTweets(data)
        self.themeModels(data, n_components_in, n_top_words_in)
        
    def weeklyTextAnalysis(self, data, n_components_in, n_top_words_in):
        print("**************************************************************************************", "\n")
        for week in list(data.week_number.unique()):
            df = data[data.week_number == week]
            print("Para la semana " + str(week) + " del año obtenemos: " + "\n")
            self.tweetsAnalysis(df, n_components_in, n_top_words_in)
            print("**************************************************************************************", "\n")