import re
import sys
import math


class InvertedIndex:
    """ A simple inverted index, as explained in Lecture 1. """

    def __init__(self):
        """ Create an empty inverted index. """

        self.inverted_lists = {}
        self.document_length = {}

    def read_from_file(self, file_name, bm25_k, bm25_b):
        """ Construct from given file (one record per line).
        >>> ii = InvertedIndex()
        >>> ii.read_from_file("example.txt", 1.75, 0.75)
        >>> sorted(ii.inverted_lists.items())
        [('docum', [(1, 0.0), (2, 0.0), (3, 0.0)]), ('first', [(1, 1.885)]), \
('second', [(2, 2.325)]), ('third', [(3, 2.521)])]
         """

        record_id = 0
        word_count = 0
        with open(file_name) as file:
            for line in file:
                record_id += 1
                self.document_length[record_id] = 0
                for word in re.split("\W+", line):
                    word = word.lower()
                    if len(word) > 0:
                        self.document_length[record_id] += 1
                        # If word seen first time, create inverted list.
                        if word not in self.inverted_lists:
                            self.inverted_lists[word] = []
                        # Append record id to inverted list.
                        if (len(self.inverted_lists[word]) == 0 or
                                self.inverted_lists[word][-1][0] != record_id):
                            self.inverted_lists[word].append((record_id, 1))
                        else:
                            count = self.inverted_lists[word][-1][1] + 1
                            self.inverted_lists[word][-1] = (record_id, count)

                word_count += self.document_length[record_id]
        total_doc = record_id
        avdl = word_count/total_doc
        self.tf_star(total_doc, avdl, bm25_k, bm25_b)

    def tf_star(self, total_doc, avdl, k, b):

        for word in self.inverted_lists:
            df = len(self.inverted_lists[word])
            for i, record in enumerate(self.inverted_lists[word]):
                tf = self.inverted_lists[word][i][1]
                doc_index = self.inverted_lists[word][i][0]
                tf_star = (tf * (k + 1) / (k * (1 - b + b *
                           (self.document_length[doc_index] / avdl)) + tf))
                idf = math.log((total_doc / df), 2)
                bm25_score = round(tf_star * idf, 3)
                self.inverted_lists[word][i] = (doc_index, bm25_score)

    def merge(self, list1, list2):
        """ Merging two inverted lists.
        >>> ii = InvertedIndex()
        >>> ii.merge([(2, 0), (5, 2), (7, 7), (8, 6)], \
                     [(4, 1), (5, 3), (6, 3), (8, 3), (9, 8)])
        [(2, 0), (4, 1), (5, 5), (6, 3), (7, 7), (8, 9), (9, 8)]
        """

        ret = []
        i = 0
        j = 0
        while(i < len(list1) and j < len(list2)):
            if list1[i][0] < list2[j][0]:
                ret.append((list1[i]))
                i += 1
            elif list1[i][0] > list2[j][0]:
                ret.append((list2[j]))
                j += 1
            else:
                ret.append((list1[i][0], list1[i][1] + list2[j][1]))
                i += 1
                j += 1
                if(i == len(list1)):
                    ret.extend((list2[j:]))
                elif(j == len(list2)):
                    ret.extend((list1[i:]))
        return ret

    def intersect(self, list1, list2):
        """ Intersection of two inverted lists. """

        ret = []
        i = 0
        j = 0
        while(i < len(list1) and j < len(list2)):
            if list1[i] < list2[j]:
                i += 1
            elif list1[i] > list2[j]:
                j += 1
            else:
                ret.append(list1[i])
                i += 1
                j += 1
        return ret

    def process_query(self, qry):
        """ Merging two inverted lists.
        >>> ii = InvertedIndex()
        >>> ii.read_from_file("example.txt", 1.75, 0.75)
        >>> ii.process_query("docum third")
        [(3, 2.521), (1, 0.0), (2, 0.0)]
        """

        keywords = list(filter(None, re.split("\W+", qry.lower())))
        ret = []
        if len(keywords) > 0 and keywords[0] in self.inverted_lists:
            ret = self.inverted_lists[keywords[0]]

        for keyword in keywords[1:]:
            if keyword in self.inverted_lists:
                ret = self.merge(ret, self.inverted_lists[keyword])

        ret = sorted(ret, key=lambda x: x[1], reverse=True)
        return ret

    def render_output(self, file_name, qry, qry_res, max_res):
        """
        (Exercise 01-1-4)
        Output results. Load documents from HD to save memory.
        """

        outputted = 0
        with open(file_name) as file:
            for i, line in enumerate(file):
                for index in qry_res[:max_res]:
                    if i + 1 == index[0]:
                        outputted += 1
                        print(re.sub('\\b(' + '|'.join(qry) + ')\\b',
                                     "\033[3;37;40m" + '\\1' + "\033[0;0m",
                                     line, flags=re.IGNORECASE))
                    if outputted >= max_res:
                        """ Look no further after we reached the desired number
                        of results. """
                        break


class EvaluateBenchmark:

    def __init__(self):
        """ Create an empty dict"""
        self.inverted_lists = {}

    def precision_at_k(self, result_ids, relevant_ids, k):
        """ For Evaluating P@K
        >>> eb = EvaluateBenchmark()
        >>> eb.precision_at_k([0, 1, 2, 5, 6], [0, 2, 5, 6, 7, 8], 4)
        0.75
        """

        correct = 0
        for result in result_ids[:k]:
            if result in relevant_ids:
                correct += 1
        return round(correct/k, 3)

    def average_precision(self, result_ids, relevant_ids):
        """ For Evaluating AP
        >>> eb = EvaluateBenchmark()
        >>> eb.average_precision([582, 17, 5666, 10003, 10], \
                                 [10, 582, 877, 10003])
        0.525
        """
        index_list = []
        ap = 0
        for i, result in enumerate(result_ids):
            if result in relevant_ids:
                index_list.append(i+1)
        for i, item in enumerate(index_list):
            tmp = index_list[:i+1]
            ap += (len(tmp) / item) / len(relevant_ids)
        return ap

    def evaluate_benchmark(self, file_name):
        record_id = 0
        p_3 = 0
        p_r = 0
        ap = 0
        mp_3 = 0
        mp_r = 0
        map_r = 0
        with open(file_name) as file:
            for line in file:
                record_id += 1
                item = re.split(r'\t+', line)
                words = item[0]
                for index in item[1].split():
                    if words not in self.inverted_lists:
                        self.inverted_lists[words] = []
                    self.inverted_lists[words].append(index)
            for qry in self.inverted_lists:
                result_ids = [result[0] for result in ii.process_query(qry)]
                relevant_ids = list(map(int, sorted(self.inverted_lists[qry])))
                p_3 += self.precision_at_k(result_ids, relevant_ids,  3)
                p_r += self.precision_at_k(result_ids, relevant_ids,
                                           len(relevant_ids))
                ap += self.average_precision(result_ids, relevant_ids)
            mp_3 = p_3 / len(self.inverted_lists)
            mp_r = p_r / len(self.inverted_lists)
            map_r = ap / len(self.inverted_lists)
            print("MP@3 : ", mp_3)
            print("MP@R : ", mp_r)
            print("MAP : ", map_r)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 inverted_index.py <file>")
        sys.exit()
    file_name = sys.argv[1]
    ii = InvertedIndex()
    ii.read_from_file(file_name, 1, .1)
    eb = EvaluateBenchmark()
    eb.evaluate_benchmark("movies-benchmark.txt")
    while (True):
        qry = input("Enter query: ")
        """ Use the same word matching approach (regex on "\W+") as above.
        Filter out empty (None) strings."""
        keywords = list(filter(None, re.split("\W+", qry.lower())))
        print()
        ii.render_output(file_name, keywords, ii.process_query(qry), 3)
