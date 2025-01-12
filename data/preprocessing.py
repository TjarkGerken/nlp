import re
import nltk
import string
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, label_binarize

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

class DataProvider():   
    def __init__(self):
        self.data = self.__get_data()
        
    def __get_data(self): 
        """
        Get preprocessed data with default pipeline.
        
        Login using e.g. `huggingface-cli login` to access this dataset
        """
        
        df = pd.read_csv("hf://datasets/newsmediabias/fake_news_elections_labelled_data/cleaned_fakenewsdata.csv")

        tokenizer = NLTKTokenizer()
        df['cleaned_text'] = df['text'].apply(tokenizer.transform)
        
        return df
    
    def get_raw_datasets(self):
        y = DataProvider.map_labels(self.data['label'])
        
        return DataProvider.split_data(self.data['cleaned_text'], y)
    
    def get_bow_datasets(self):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(self.data['cleaned_text'])
        y = DataProvider.map_labels(self.data['label'])
        
        return DataProvider.split_data(X, y)
    
    def get_tfidf_datasets(self):        
        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform(self.data['cleaned_text'])
        y = DataProvider.map_labels(self.data['label'])
        
        return DataProvider.split_data(X, y)
    
    @staticmethod
    def split_data(X, y):
        #scaler = StandardScaler(with_mean=False)
        scaler = MaxAbsScaler()
        X = scaler.fit_transform(X)
        
        return train_test_split(X, y, test_size=0.2, random_state=7)
    
    @staticmethod
    def map_labels(labels):
        return labels.map({'FAKE': 0, 'REAL': 1})
        
class NLTKTokenizer():
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def transform(self, text):
        # Tokenization
        #return text.apply(self.tokenize)
        return self.tokenize(text)

    def tokenize(self, text):
        # Normalization
        text = re.sub(r'\W', ' ', str(text))  # only alphanumeric characters
        
        tokens = word_tokenize(text.lower())  
        
        # Stopwords 
        filtered_tokens = [t for t in tokens if t not in self.stop_words and t not in string.punctuation]

        # Lemmatization (+ POS-Tagging)
        lemmatized_tokens = [self.lemmatize_with_pos(token, pos) for token, pos in pos_tag(filtered_tokens)]
        
        return ' '.join(lemmatized_tokens)

    def lemmatize_with_pos(self, token, pos):
        # convert NLTK-POS-Tags in WordNet-POS-Tags
        pos = self.get_wordnet_pos(pos)
        return self.lemmatizer.lemmatize(token, pos=pos) if pos else self.lemmatizer.lemmatize(token)

    def get_wordnet_pos(self, nltk_pos):
        # convert NLTK-POS-Tags in WordNet-POS-Tags
        if nltk_pos.startswith('J'):
            return 'a'  # adjectives
        elif nltk_pos.startswith('V'):
            return 'v'  # verbs
        elif nltk_pos.startswith('N'):
            return 'n'  # nouns
        elif nltk_pos.startswith('R'):
            return 'r'  # adverbs
        else:
            return None