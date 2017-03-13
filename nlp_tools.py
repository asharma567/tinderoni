from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def special_char_translation(doc):
    from unidecode import unidecode
    return ' '.join([unidecode(word) for word in doc.split()])

def _unescape(text):
    import re, htmlentitydefs
    def fixup(m):
        text = m.group(0)
        if text[:2] == "&#":
            # character reference
            try:
                if text[:3] == "&#x":
                    return unichr(int(text[3:-1], 16))
                else:
                    return unichr(int(text[2:-1]))
            except ValueError:
                pass
        else:
            # named entity
            try:
                text = unichr(htmlentitydefs.name2codepoint[text[1:-1]])
            except KeyError:
                pass
        return text # leave as is
    return re.sub("&#?\w+;", fixup, text)

def instantiate_tfv(ngrams=(1,3), top_n_features=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.corpus import stopwords
    '''
    
    '''
    tfidf = TfidfVectorizer(
                        analyzer='word',
                        stop_words=set(stopwords.words('english')),
                        sublinear_tf=True,
                        ngram_range=ngrams,
                        smooth_idf=True,
                        max_features=top_n_features
                        )
    return tfidf

def NMF(corpus, n_topics, n_top_words):
    from sklearn import decomposition
    from time import time


    #timing the clustering process
    t0 = time()
    tfidf = instantiate_tfv()

    #NMF for grouped comments
    tfidf_vectorized_corpus = tfidf.fit_transform(corpus)

    vocab = tfidf.get_feature_names()
    nmf = decomposition.NMF(n_components=n_topics).fit(tfidf_vectorized_corpus)

    print("done in %0.3fs." % (time() - t0))
    for topic_idx, topic in enumerate(nmf.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join( [vocab[i] for i in topic.argsort()[:-n_top_words - 1:-1]] ))        

def unescape(text):
    import re, htmlentitydefs
    def fixup(m):
        text = m.group(0)
        if text[:2] == "&#":
            # character reference
            try:
                if text[:3] == "&#x":
                    return unichr(int(text[3:-1], 16))
                else:
                    return unichr(int(text[2:-1]))
            except ValueError:
                pass
        else:
            # named entity
            try:
                text = unichr(htmlentitydefs.name2codepoint[text[1:-1]])
            except KeyError:
                pass
        return text # leave as is
    return re.sub("&#?\w+;", fixup, text)

def remove_stop_words(doc):
    from nltk.corpus import stopwords
    stopwords = set(stopwords.words('english'))
    return ' '.join([word for word in doc.split() if word.lower() not in stopwords])

ABREVIATIONS_DICT = {
    "'m":' am',
    "'ve":' have',
    "'ll":" will",
    "'d":" would",
    "'s":" is",
    "'re":" are",
    "  ":" ",
    "' s": " is",
}

def multiple_replace(text, adict=ABREVIATIONS_DICT):
    import re
    '''
    Does a multiple find/replace
    '''
    rx = re.compile('|'.join(map(re.escape, adict)))
    def one_xlat(match):
        return adict[match.group(0)]
    return rx.sub(one_xlat, text.lower())


def find_stop_words(corpus):
    '''
    takes in a normalized corpus and returns stop words in pandas Series
    '''
    unpacked_list = [word for document in corpus for word in document.split()]
    
    return pd.Series(unpacked_list).value_counts()


#I question the need for this but lets just do it for now
def _multiple_replace(text, adict=ABREVIATIONS_DICT):
    import re
    '''
    Does a multiple find/replace
    '''
    rx = re.compile('|'.join(map(re.escape, adict)))
    def one_xlat(match):
        return adict[match.group(0)]
    return rx.sub(one_xlat, text.lower())

def _special_char_translation(doc):
    from unidecode import unidecode

    return ' '.join([unidecode(word) for word in doc.split()])

def _remove_stop_words(doc):
    from nltk.corpus import stopwords
    stopwords_set = set(stopwords.words('english'))    
    return ' '.join([word for word in doc.split() if word.lower() not in stopwords_set])

def normalize(document, post_normalization_stop_words={}): # this is bad design it should be none
    from string import punctuation
    from nltk.stem.snowball import SnowballStemmer
    from nltk.stem.wordnet import WordNetLemmatizer

    WHITE_SPACE = ' '

    #this isn't ideal from an nlp standpoint because it affects normalization
    decoded_doc = _special_char_translation(document.decode("utf8"))
    abbreviations_removed_doc = _multiple_replace(decoded_doc)
    stops_removed_doc = _remove_stop_words (abbreviations_removed_doc)
    punc_removed = ''.join([char for char in stops_removed_doc if char not in set(punctuation)])    

    snowball = SnowballStemmer('english')
    wordnet = WordNetLemmatizer()
    
    stripped_lemmatized = map(wordnet.lemmatize, punc_removed.split())
    stripped_lemmatized_stemmed = map(snowball.stem, stripped_lemmatized)
    
    return WHITE_SPACE.join([word for word in stripped_lemmatized_stemmed if word not in post_normalization_stop_words])