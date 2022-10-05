from collections import defaultdict
from string import punctuation

from lxml import etree
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


# import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('averaged_perceptron_tagger')

class KeyTermExtraction:

    def __init__(self, file_path: str):
        self.file_path = file_path
        stop_words = stopwords.words("english") + list(punctuation)
        self.words_to_remove = defaultdict(bool, {word: True for word in stop_words})

    def _get_xml_content(self):
        return etree.parse(self.file_path).getroot()

    def _lemmatize(self, words):
        lemmatizer = WordNetLemmatizer()
        news_words = [lemmatizer.lemmatize(word, pos='n') for word in words]
        return news_words

    def _clean(self, words):
        cleaned_words = [word for word in words if not self.words_to_remove[word] and pos_tag([word])[0][1] == "NN"]
        return cleaned_words

    def parse(self):
        texts = []
        heads = []
        root = self._get_xml_content()
        for news_tag in root[0]:
            news_head = ""
            news_text = ""
            for news_item in news_tag:
                news_item_name = news_item.get("name")
                if "head" == news_item_name:
                    news_head = news_item.text
                if "text" == news_item_name:
                    news_text = news_item.text
            if news_head and news_text:
                news_words = word_tokenize(news_text.lower())
                news_words = self._lemmatize(news_words)
                news_words = self._clean(news_words)
                heads.append(news_head)
                texts.append(" ".join(news_words))
        return tuple([heads, texts])

    def tf_ifd(self, texts):
        result = []
        vectorizer = TfidfVectorizer(input='content', use_idf=True, lowercase=True, analyzer='word', ngram_range=(1, 1),
                                     stop_words=None)
        weighted_matrix = vectorizer.fit_transform(texts).toarray()
        terms = vectorizer.get_feature_names()
        for text_index, text in enumerate(texts):
            text_data = list(zip(terms, weighted_matrix[text_index]))
            text_data = sorted(text_data, key=lambda word: (word[1], word[0]), reverse=True)
            result.append(text_data)
        return result


def print_freq_words(heads, news_lst, n):
    for head_idx, head in enumerate(heads):
        print(f"{head}:")
        words = news_lst[head_idx][:n]
        print(*[word[0] for word in words], "\n")


def main():
    key_term = KeyTermExtraction("../data/news.xml")
    heads, texts = key_term.parse()
    news_lst = key_term.tf_ifd(texts)
    print_freq_words(heads, news_lst, 5)


if __name__ == "__main__":
    main()
