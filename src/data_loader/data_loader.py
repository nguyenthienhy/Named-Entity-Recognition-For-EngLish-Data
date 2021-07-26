from itertools import islice
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from src.utils.get_type_word import *
from tqdm import tqdm


class DataLoader:
    def __init__(self):
        pass

    @staticmethod
    def load_and_clean_data(file_path):
        all_sentences, named_entities = [], []
        with open(file_path) as train_file:
            words, entities, unique_entities = [], [], set()
            for line in tqdm(islice(train_file, 1, None)):
                word = line.split(' ')[0]
                named_entity = line.split(' ')[-1].strip('\n')
                if line in ('\n', '\r\n'):
                    # end of a sentence
                    all_sentences.append(' '.join(words))
                    named_entities.append(' '.join(entities))
                    unique_entities |= set(entities)
                    words, entities = [], []
                else:
                    if word:
                        # Performing Word Lemmatization on text
                        word_lemmatizer = WordNetLemmatizer()
                        word, type = nltk.pos_tag(word_tokenize(word))[0]
                        type = get_wordnet_pos(type)
                        if type:
                            lemmatized_word = word_lemmatizer.lemmatize(word, type)
                        else:
                            lemmatized_word = word_lemmatizer.lemmatize(word)
                        words.append(lemmatized_word)
                        entities.append(named_entity)

        return all_sentences, named_entities, unique_entities


if __name__ == "__main__":
    app = DataLoader()
    app.load_and_clean_data("C:\\Users\\hyngu\\Desktop\\NER\\data\\test.txt")
