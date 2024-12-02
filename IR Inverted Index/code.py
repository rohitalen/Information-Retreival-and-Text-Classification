import xml.etree.ElementTree as ET
from nltk.stem import PorterStemmer
import math
import re


# Collect stopwords from file
def load_stopwords(filepath):
    stopwords = set()
    with open(filepath, 'r') as file:
        for word in file:
            stopwords.add(word.strip().lower())
    return stopwords

# Process document content: tokenization, stemming, stopword removal
def tokenize_and_clean(content, stopwords, stemmer):
    pseudo_tokens = re.findall(r"\d+\.\d+(?:bn|m)?|[\w']+", content.lower())
    filtered_tokens = [word for word in pseudo_tokens if word not in stopwords]
    joined_tokens = ' '.join(filtered_tokens)
    final_tokens = re.findall(r'\d+\.\d+(?:bn|m)?|\w+', joined_tokens)
    stemmed_tokens = [stemmer.stem(token) for token in final_tokens if token not in stopwords]
    return stemmed_tokens

# Preprocess XML document file and build inverted index
def build_inverted_index(doc_file, stopwords):
    inverted_index = {}
    document_ids = set()
    stemmer = PorterStemmer()

    tree = ET.parse(doc_file)
    root = tree.getroot()

    for doc in root.findall('DOC'):
        doc_id = doc.find('DOCNO').text.strip()
        title = doc.find('HEADLINE').text.strip() if doc.find('HEADLINE') is not None else ""
        body = doc.find('TEXT').text.strip() if doc.find('TEXT') is not None else ""

        combined_text = f"{title} {body}"
        processed_tokens = tokenize_and_clean(combined_text, stopwords, stemmer)

        document_ids.add(doc_id)

        for idx, token in enumerate(processed_tokens):
            if token in inverted_index:
                if doc_id in inverted_index[token][1]:
                    inverted_index[token][1][doc_id].append(idx)
                else:
                    inverted_index[token][1][doc_id] = [idx]
                    inverted_index[token][0] += 1
            else:
                inverted_index[token] = [1, {doc_id: [idx]}]

    return inverted_index, document_ids

# Perform proximity or phrase search
def perform_proximity_search(index, first_word, second_word, gap, is_phrase=False):
    first_word_docs = index.get(first_word, None)
    second_word_docs = index.get(second_word, None)

    if first_word_docs is None or second_word_docs is None:
        return set()

    first_docs = first_word_docs[1]
    second_docs = second_word_docs[1]

    common_docs = set(first_docs.keys()).intersection(second_docs.keys())
    matched_docs = set()

    for doc_id in common_docs:
        first_positions = first_docs[doc_id]
        second_positions = second_docs[doc_id]
        found = False

        for pos1 in first_positions:
            for pos2 in second_positions:
                if is_phrase:
                    if pos2 - pos1 == gap:
                        found = True
                        matched_docs.add(doc_id)
                        break
                elif abs(pos1 - pos2) <= gap:
                    found = True
                    matched_docs.add(doc_id)
                    break
            if found:
                break

    return matched_docs

# Boolean search
def boolean_search(index, token):
    if token in index:
        return set(index[token][1].keys())
    else:
        return set()

# Parse and process a query (boolean or phrase)
def parse_query(query, index, doc_set, stopwords, stemmer):
    query_id, query_text = query.split(' ', 1)

    # Check for proximity search pattern
    proximity_match = re.match(r'^#(\d+)\(\s*([^,]+)\s*,\s*([^,]+)\s*\)$', query_text)

    if proximity_match:
        # Extract distance and terms for proximity search
        distance = int(proximity_match.group(1))
        term_1, term_2 = proximity_match.group(2), proximity_match.group(3)
        # Perform proximity search
        doc_ids = perform_proximity_search(index, stemmer.stem(term_1.lower()), stemmer.stem(term_2.lower()), distance, False)
    else:
        tokens = query_text.split()
        # Handle boolean operators (AND/OR)
        and_pos, or_pos = tokens.index('AND') if 'AND' in tokens else -1, tokens.index('OR') if 'OR' in tokens else -1
        main_operator_pos = max(and_pos, or_pos)

        if main_operator_pos != -1:
            left_side, right_side = tokens[:main_operator_pos], tokens[main_operator_pos + 1:]

            # Process left side
            left_docs = process_query_side(left_side, index, stopwords, stemmer, doc_set)

            # Process right side
            right_docs = process_query_side(right_side, index, stopwords, stemmer, doc_set)

            # Perform intersection for AND or union for OR
            doc_ids = left_docs.intersection(right_docs) if and_pos != -1 else left_docs.union(right_docs)
        else:
            # Simple query without AND/OR
            doc_ids = process_query_side(tokens, index, stopwords, stemmer, doc_set)

    return query_id, doc_ids


def process_query_side(tokens, index, stopwords, stemmer, doc_set):
    if tokens[0] == 'NOT':
        doc_ids = phrase_or_boolean_search(tokens[1:], index, stopwords, stemmer)
        return doc_set.difference(doc_ids)
    return phrase_or_boolean_search(tokens, index, stopwords, stemmer)

def phrase_or_boolean_search(tokens, index, stopwords, stemmer):
    cleaned_tokens = tokenize_and_clean(' '.join(tokens), stopwords, stemmer)
    if len(cleaned_tokens) == 1:
        return boolean_search(index, cleaned_tokens[0])
    else:
        return perform_proximity_search(index, cleaned_tokens[0], cleaned_tokens[1], 1, True)

# Perform ranked search using TF-IDF
def ranked_tfidf_search(query, index, doc_set, stopwords, stemmer):
    query_id, query_text = query.split(' ', 1)
    query_terms = tokenize_and_clean(query_text, stopwords, stemmer)
    tfidf_scores = {}
    for doc in doc_set:
        tfidf_scores[doc] = 0
    for term in query_terms:
        if term in index:
            for doc in doc_set:
                if doc in index[term][1]:
                    term_freq = 1 + math.log(len(index[term][1][doc]), 10)
                    inv_doc_freq = math.log(len(doc_set) / index[term][0], 10)
                    tfidf_scores[doc] += term_freq * inv_doc_freq

    return query_id, sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)

# Write ranked search results to file
def write_ranked_results(query_file, index, doc_set, stopwords, stemmer):
    with open(query_file, 'r') as query_f, open('results.ranked.txt', 'w') as result_f:
        for query in query_f:
            query_id, ranked_docs = ranked_tfidf_search(query.strip(), index, doc_set, stopwords, stemmer)
            for i in range(len(ranked_docs)):
                if i< 150:
                    result_f.write(str(query_id) + ',' + str(ranked_docs[i][0]) + ',' + str(round(ranked_docs[i][1],4)) + '\n')

# Write inverted index to a file
def write_inverted_index(index):
    with open('index.txt', 'w') as f:
        for token, (doc_freq, docs) in index.items():
            f.write(f"{token}:{doc_freq}\n")
            for doc_id, positions in docs.items():
                f.write(f"\t{doc_id}:{','.join(map(str, positions))}\n")

# Write boolean search results to file
def write_boolean_results(query_file, index, doc_set, stopwords, stemmer):
    with open(query_file, 'r') as query_f, open('results.boolean.txt', 'w') as result_f:
        for query in query_f:
            query_id, doc_ids = parse_query(query.strip(), index, doc_set, stopwords, stemmer)
            for doc_id in doc_ids:
                result_f.write(str(query_id) + "," + str(doc_id) + '\n')

# Main function to run the search engine
def main(stopwords_file, doc_file, boolean_query_file, ranked_query_file):
    stemmer = PorterStemmer()
    stopwords = load_stopwords(stopwords_file)

    inverted_index, doc_set = build_inverted_index(doc_file, stopwords)

    write_inverted_index(inverted_index)
    write_boolean_results(boolean_query_file, inverted_index, doc_set, stopwords, stemmer)
    write_ranked_results(ranked_query_file, inverted_index, doc_set, stopwords, stemmer)

if __name__ == "__main__":
    main('ttds_2023_english_stop_words.txt', 'trec.5000.xml', 'queries.boolean.txt', 'queries.ranked.txt')
 # type: ignore
                                 