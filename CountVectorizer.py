from math import log


class CountVectorizer:
    """Convert a collection of text documents to a matrix of token counts"""

    def __init__(self):
        self.words = {}

    def fit_transform(self, corpus: list[str]) -> list[list[int]]:
        """Learn the vocabulary dictionary and return document-term matrix"""
        order = 0
        for elem in corpus:
            word_of_sent = elem.lower().split()
            for word in word_of_sent:
                if word not in self.words:
                    self.words[word] = order
                    order += 1

        matrix = []
        for sent in corpus:
            row = [0] * len(self.words)
            for key_word in self.words:
                row[self.words[key_word]] = sent.lower().count(key_word)

            matrix.append(row)

        return matrix

    def get_feature_names(self) -> list[str]:
        """Get output feature names for transformation"""
        return [k for k, v in sorted(self.words.items(), key=lambda item: item[1])]


class TfidfTransformer:
    def tf_transform(self, matrix) -> list[list[int]]:
        tf_matrix = []
        for words_list in matrix:
            tf_vector = []
            for word_count in words_list:
                tf = round(word_count / sum(words_list), 3)
                tf_vector.append(tf)
            tf_matrix.append(tf_vector)
        return tf_matrix

    def idf_transform(self, matrix) -> list[int]:
        idf_matrix = []
        words = [0] * len(matrix[0])
        for vector in matrix:
            for i, count in enumerate(vector):
                if count != 0:
                    words[i] += 1
        for word in words:
            result = log((len(matrix) + 1) / (word + 1)) + 1
            idf_matrix.append(round(result, 3))
        return idf_matrix

    def fit_transform(self, matrix):
        tf = self.tf_transform(matrix)
        idf = self.idf_transform(matrix)
        mult_matrix = []
        for tf_vector in tf:
            mult_vector = []
            for tf_i, idf_i in zip(tf_vector, idf):
                mult_vector.append(round(tf_i * idf_i, 3))
                mult_matrix.append(mult_vector)
        return mult_matrix


class TfidfVectorizer(CountVectorizer):
    def __init__(self):
        super().__init__()
        self.tfidf_transformer = TfidfTransformer()

    def fit_transform(self, corpus):
        count_matrix = super().fit_transform(corpus)
        result = self.tfidf_transformer.fit_transform(count_matrix)
        return result


if __name__ == "__main__":
    corpus = [
        "Crock Pot Pasta Never boil pasta again",
        "Pasta Pomodoro Fresh ingredients Parmesan to taste",
    ]
    vectorizer = TfidfVectorizer
    tfidf_matrix = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names())
    print(tfidf_matrix)
