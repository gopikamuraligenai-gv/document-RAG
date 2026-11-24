
import re
import math
from collections import Counter


class TFIDFRetriever:

    def __init__(self, documents, k=5):
        self.documents = documents
        self.k = k

        self.texts = [
            self._get_text(doc) for doc in documents
        ]

        self.tokenized_docs = [
            self._tokenize(text) for text in self.texts
        ]

        self.idf = self._calculate_idf()

        self.document_vectors = [
            self._create_vector(tokens)
            for tokens in self.tokenized_docs
        ]

    def _get_text(self, document):

        if hasattr(document, "page_content"):
            return document.page_content

        return str(document)

    def _tokenize(self, text):

        return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

    def _calculate_idf(self):

        total_docs = len(self.tokenized_docs)
        document_frequency = Counter()

        for tokens in self.tokenized_docs:
            document_frequency.update(set(tokens))

        idf = {}

        for word, frequency in document_frequency.items():
            idf[word] = math.log(
                (1 + total_docs) / (1 + frequency)
            ) + 1

        return idf

    def _create_vector(self, tokens):

        term_frequency = Counter(tokens)
        total_terms = len(tokens)

        if total_terms == 0:
            return {}

        vector = {}

        for word, count in term_frequency.items():

            tf = count / total_terms
            idf = self.idf.get(word, 0)

            vector[word] = tf * idf

        return vector

    def _cosine_similarity(self, vector1, vector2):

        common_words = set(vector1) & set(vector2)

        dot_product = sum(
            vector1[word] * vector2[word]
            for word in common_words
        )

        norm1 = math.sqrt(
            sum(value ** 2 for value in vector1.values())
        )

        norm2 = math.sqrt(
            sum(value ** 2 for value in vector2.values())
        )

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def retrieve(self, query):

        query_tokens = self._tokenize(query)
        query_vector = self._create_vector(query_tokens)

        scores = []

        for index, document_vector in enumerate(
            self.document_vectors
        ):

            score = self._cosine_similarity(
                query_vector,
                document_vector
            )

            scores.append((index, score))

        ranked_results = sorted(
            scores,
            key=lambda item: item[1],
            reverse=True
        )

        results = []

        for index, score in ranked_results[:self.k]:

            if score > 0:
                results.append({
                    "document": self.documents[index],
                    "score": score
                })

        return results
