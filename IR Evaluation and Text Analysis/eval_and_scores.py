import pandas as pd
import numpy as np
import re
from gensim.corpora.dictionary import Dictionary # type: ignore
from gensim.models import LdaModel # type: ignore
from nltk.stem import PorterStemmer
# Dictionaries to store system results and relevance judgments
system_results = {}
query_relevance = {}
corpus_data = {}
stemmer = PorterStemmer()
stop_words = []
with open('stop_words.txt', 'r', encoding='UTF-8-sig') as f:
    for line in f:
        stop_words.append(line.replace('\n', ''))
f.close()

def load_data_task1():
    """Reads and processes input CSV files."""
    system_data = pd.read_csv('ttdssystemresults.csv')
    relevance_data = pd.read_csv('qrels.csv')
    
    # Populate system results dictionary
    for i in range(len(system_data)):
        system_id = system_data['system_number'][i]
        query_id = system_data['query_number'][i]
        doc_details = [
            system_data['doc_number'][i],
            system_data['rank_of_doc'][i],
            system_data['score'][i]
        ]
        
        if system_id not in system_results:
            system_results[system_id] = {}
        system_results[system_id].setdefault(query_id, []).append(doc_details)
    
    # Populate query relevance dictionary
    for i in range(len(relevance_data)):
        query_id = relevance_data['query_id'][i]
        relevance_info = [
            relevance_data['doc_id'][i],
            relevance_data['relevance'][i]
        ]
        query_relevance.setdefault(query_id, []).append(relevance_info)

def calculate_precision(retrieved_docs, query_id, cutoff):
    """Calculates the number of relevant documents within the top 'cutoff' retrieved documents."""
    relevant_count = 0
    for doc in retrieved_docs[:cutoff]:
        for relevant_doc in query_relevance[query_id]:
            if doc[0] == relevant_doc[0]:
                relevant_count += 1
                break
    return relevant_count

def get_relevance_grades(k, retrieved_docs, query_id):
    """Generates relevance grades for the top 'k' retrieved documents."""
    grades = [0] * k
    for i in range(k):
        for relevant_doc in query_relevance[query_id]:
            if retrieved_docs[i][0] == relevant_doc[0]:
                grades[i] = relevant_doc[1]
                break
    return grades

def calculate_dcg(relevance_grades, k):
    """Calculates the Discounted Cumulative Gain (DCG)."""
    dcg = relevance_grades[0]
    for i in range(2, k + 1):
        dcg += relevance_grades[i - 1] / np.log2(i)
    return dcg

def get_ideal_relevance(k, query_id):
    """Generates an ideal relevance grade distribution for a query."""
    ideal_grades = [rel[1] for rel in query_relevance[query_id]]
    ideal_grades.extend([0] * (k - len(ideal_grades)))
    return sorted(ideal_grades, reverse=True)[:k]

def EVAL():
    """Calculates evaluation metrics for all systems and writes results to ir_eval.csv."""
    results = []  # List to collect results for all systems and queries

    load_data_task1()
    precision_cutoff = 10
    recall_cutoff = 50

    for system_id in range(1, 7):
        precision_scores = []
        recall_scores = []
        r_precision_scores = []
        average_precision_scores = []
        ndcg_scores = {10: [], 20: []}

        for query_id in range(1, 11):
            retrieved_docs = system_results[system_id][query_id]

            # Precision and Recall
            p_at_10 = calculate_precision(retrieved_docs, query_id, precision_cutoff) / precision_cutoff
            precision_scores.append(p_at_10)

            total_relevant_docs = len(query_relevance[query_id])
            r_at_50 = calculate_precision(retrieved_docs, query_id, recall_cutoff) / total_relevant_docs
            recall_scores.append(r_at_50)

            # r-Precision
            r_prec = calculate_precision(retrieved_docs, query_id, total_relevant_docs) / total_relevant_docs
            r_precision_scores.append(r_prec)

            # Average Precision
            ap = 0
            for i, doc in enumerate(retrieved_docs):
                if any(doc[0] == rel_doc[0] for rel_doc in query_relevance[query_id]):
                    ap += calculate_precision(retrieved_docs, query_id, i + 1) / (i + 1)
            avg_prec = ap / total_relevant_docs
            average_precision_scores.append(avg_prec)

            # nDCG Calculations
            ndcg_10 = 0
            ndcg_20 = 0
            for cutoff in [10, 20]:
                relevance_grades = get_relevance_grades(cutoff, retrieved_docs, query_id)
                dcg = calculate_dcg(relevance_grades, cutoff)
                ideal_grades = get_ideal_relevance(cutoff, query_id)
                ideal_dcg = calculate_dcg(ideal_grades, cutoff)
                ndcg = dcg / ideal_dcg if ideal_dcg else 0
                if cutoff == 10:
                    ndcg_10 = ndcg
                else:
                    ndcg_20 = ndcg
                ndcg_scores[cutoff].append(ndcg)

            # Append individual query results
            results.append({
                "system_number": system_id,
                "query_number": query_id,
                "P@10": round(p_at_10, 3),
                "R@50": round(r_at_50, 3),
                "r-precision": round(r_prec, 3),
                "AP": round(avg_prec, 3),
                "nDCG@10": round(ndcg_10, 3),
                "nDCG@20": round(ndcg_20, 3)
            })

        # Append mean results for the system
        results.append({
            "system_number": system_id,
            "query_number": "mean",
            "P@10": round(np.mean(precision_scores), 3),
            "R@50": round(np.mean(recall_scores), 3),
            "r-precision": round(np.mean(r_precision_scores), 3),
            "AP": round(np.mean(average_precision_scores), 3),
            "nDCG@10": round(np.mean(ndcg_scores[10]), 3),
            "nDCG@20": round(np.mean(ndcg_scores[20]), 3)
        })

    # Convert results to a DataFrame and save as CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("ir_eval.csv", index=False)

def processText(text):
    result = [stemmer.stem(word) for word in re.findall(r'\w+', text) if word.lower() not in stop_words]
    return result


# Function to read the TSV file and process the verses
def load_data():
    file_data = pd.read_csv('bible_and_quran.tsv', sep='\t', header=None)
    full_text = ''
    
    for idx in range(len(file_data)):
        corpus_name = file_data[0][idx]
        verse_text = file_data[1][idx]

        if corpus_name in corpus_data:
            corpus_data[corpus_name].append(verse_text)
        else:
            corpus_data.setdefault(corpus_name, []).append(verse_text)

        full_text += verse_text + '\n'

    # Process each corpus in the dictionary
    for corpus_name in corpus_data:
        corpus_data[corpus_name] = [processText(verse) for verse in corpus_data[corpus_name]]

    all_tokens = list(set(processText(full_text)))
    return all_tokens

# Function to calculate Mutual Information (MI) and Chi-squared (CHI) scores
def calculate_word_level_scores():
    corpus_tokens = load_data()
    orders = [['Quran', 'OT', 'NT'], ['OT', 'Quran', 'NT'], ['NT', 'Quran', 'OT']]
    chi_scores = []
    mi_scores = []
    for order in orders:
        target = order[0]
        targetlen = len(corpus_data[target])
        otherlen = len(corpus_data[order[1]]) + len(corpus_data[order[2]])
        N = targetlen + otherlen
        oneMI = []
        oneChi = []

        for term in corpus_tokens:
            N11 = 0
            for item in corpus_data[target]:
                if term in item:
                    N11 += 1
            N01 = targetlen - N11

            N10 = 0
            for corpora in order[1:]:
                for item in corpus_data[corpora]:
                    if term in item:
                        N10 += 1
            N00 = otherlen - N10

            N1x = N11 + N10
            Nx1 = N11 + N01
            N0x = N00 + N01
            Nx0 = N00 + N10

            sub1 = np.log2(N*N11 / (N1x*Nx1)) if N*N11 != 0 and N1x*Nx1 != 0 else 0
            sub2 = np.log2(N*N01 / (N0x*Nx1)) if N*N01 != 0 and N0x*Nx1 != 0 else 0
            sub3 = np.log2(N*N10 / (N1x*Nx0)) if N*N10 != 0 and N1x*Nx0 != 0 else 0
            sub4 = np.log2(N*N00 / (N0x*Nx0)) if N*N00 != 0 and N0x*Nx0 != 0 else 0
            mi = (N11/N)*sub1 + (N01/N)*sub2 + (N10/N)*sub3 + (N00/N)*sub4

            below = Nx1 * N1x * Nx0 * N0x
            chi = N * np.square(N11*N00-N10*N01) / below if below != 0 else 0

            oneMI.append([term, mi])
            oneChi.append([term, chi])

        mi_scores.append(sorted(oneMI, key=lambda x: x[-1], reverse=-True))
        chi_scores.append(sorted(oneChi, key=lambda x: x[-1], reverse=-True))

    print("MI")
    for each in mi_scores:
        print(each[:10])

    print("CHI")
    for each in chi_scores:
        print(each[:10])

# Function to calculate the average topic score for each corpus
def calculate_topic_scores(topic_probs, start_idx, end_idx):
    topic_avg_scores = [0] * 20  # Assume there are 20 topics
    for doc_probs in topic_probs[start_idx:end_idx]:
        for topic_id, prob in doc_probs:
            topic_avg_scores[topic_id] += prob
    return np.array(topic_avg_scores) / (end_idx - start_idx)

def HighTopicScore():
    # Combine all verses from the three corpora
    texts = corpus_data['Quran'] + corpus_data['OT'] + corpus_data['NT']
    quran_len = len(corpus_data['Quran'])
    OT_len = len(corpus_data['OT'])
    NT_len = len(corpus_data['NT'])
    lenall = len(texts)
    scoreall = []

    # Create a Gensim Dictionary and Corpus
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Train the LDA model with 20 topics
    lda = LdaModel(corpus, id2word=dictionary, num_topics=20, random_state=42, iterations=500)

    # Get the document-topic probabilities for each document
    for i in range(lenall):
        scoreall.append(lda.get_document_topics(corpus[i]))

    print('LDA Topic Distribution:')
    
    # Calculate average topic scores for each corpus (Quran, OT, NT)
    order = [[0, quran_len], [quran_len, quran_len + OT_len], [quran_len + OT_len, lenall]]
    corpus_names = ['Quran', 'OT', 'NT']
    
    avg_scores = []
    
    for i in range(3):
        avgscore = calculate_topic_scores(scoreall, order[i][0], order[i][1])
        avg_scores.append(avgscore)
        print(f"Average Topic Scores for {corpus_names[i]}:")
        print(avgscore)
    
    # Identify the topic with the highest average score in each corpus
    highest_avg_topics = []
    for i in range(3):
        highest_avg_topic = np.argmax(avg_scores[i])  # Index of the topic with the highest average score
        highest_avg_topics.append(highest_avg_topic)
        print(f"Highest average topic for {corpus_names[i]}: Topic {highest_avg_topic}")

    # Print the top 10 tokens for the highest average topic in each corpus
    print("\nTop 10 Tokens for the Highest Average Topics:")
    for i in range(3):
        highest_avg_topic = highest_avg_topics[i]
        print(f"\nTop 10 Tokens for Topic {highest_avg_topic} in {corpus_names[i]}:")
        
        # Get the words associated with this topic
        topic_words = lda.get_topic_terms(highest_avg_topic, topn=10)
        
        for term_id, prob in topic_words:
            word = dictionary[term_id]
            print(f"{word}: {prob:.4f}")

if __name__ == '__main__':
    #EVAL()
    calculate_word_level_scores()
    HighTopicScore()


'''
MI
[['god', 0.031315057358951216], ['muhammad', 0.030213244627015343], ['torment', 0.020586944429588798], ['believ', 0.020228021478441268], ['messeng', 0.01596337474190813], ['king', 0.01585793201151665], ['israel', 0.015573946813467326], ['quran', 0.01473700802932229], ['revel', 0.014482839486013775], ['unbeliev', 0.01306507415583372]]
[['jesu', 0.03865642296857681], ['israel', 0.03637950500182374], ['king', 0.03137734313453207], ['lord', 0.03071273778930124], ['ot', 0.02271016008478581], ['christ', 0.020603855836814237], ['believ', 0.018544325242376834], ['son', 0.01639084398731795], ['god', 0.016129394858056728], ['muhammad', 0.0160872814929012]]
[['jesu', 0.05662963922525349], ['christ', 0.03449451244115645], ['lord', 0.023783884082444827], ['israel', 0.015377196038873421], ['discipl', 0.015265886637701018], ['peopl', 0.011502018060738844], ['king', 0.011461225021200179], ['nt', 0.01094299833428795], ['ot', 0.010911579368191919], ['land', 0.010318888657265943]]
CHI
[['muhammad', 1667.179415512913], ['torment', 1204.0429811331248], ['believ', 1197.8308197898732], ['messeng', 944.7981741649253], ['revel', 846.7442820846965], ['quran', 814.9187417325003], ['unbeliev', 763.4216955655063], ['guidanc', 730.7404634709078], ['disbeliev', 708.9024843667341], ['deed', 660.3567554831479]]
[['jesu', 1334.8698276250661], ['christ', 709.8083088299885], ['believ', 682.3722177293454], ['ot', 631.651530007109], ['muhammad', 553.8751401887283], ['land', 518.2088078451266], ['faith', 484.01236071929293], ['torment', 465.8993361335546], ['hous', 438.1107969275539], ['receiv', 431.56552308666653]]
[['christ', 1697.6844705205801], ['discipl', 778.8954521225505], ['nt', 539.6680471448283], ['paul', 507.3512846102345], ['peter', 507.3512846102345], ['thing', 461.75895810454455], ['israel', 458.4989706404623], ['spirit', 406.49446398658444], ['peopl', 386.7692709782206], ['john', 373.19741323549994]]
LDA Topic Distribution:
Average Topic Scores for Quran:
[0.06485721 0.02342913 0.08134964 0.05665912 0.04837803 0.04286027
 0.03686216 0.03996295 0.04459008 0.03098041 0.04223253 0.02360607
 0.09537789 0.02987743 0.028581   0.06011384 0.02091999 0.07424724
 0.03859825 0.05933119]
Average Topic Scores for OT:
[0.03463955 0.04623508 0.03491948 0.03999416 0.053449   0.03117723
 0.04640523 0.0542177  0.05023581 0.06528499 0.03633742 0.05904377
 0.04401772 0.05406854 0.05604701 0.04219757 0.06226263 0.03882311
 0.05049497 0.03327596]
Average Topic Scores for NT:
[0.06393014 0.02952381 0.04833831 0.03884641 0.05038731 0.0545965
 0.02949998 0.04746812 0.06664238 0.03091288 0.03188248 0.04783027
 0.07171452 0.04583338 0.05914534 0.05052978 0.03422021 0.04587412
 0.05389479 0.03753922]
Highest average topic for Quran: Topic 12
Highest average topic for OT: Topic 9
Highest average topic for NT: Topic 12

Top 10 Tokens for the Highest Average Topics:

Top 10 Tokens for Topic 12 in Quran:
god: 0.1159
spirit: 0.0651
nation: 0.0440
prophet: 0.0410
lord: 0.0353
judg: 0.0316
jew: 0.0284
kingdom: 0.0279
faith: 0.0269
saint: 0.0265

Top 10 Tokens for Topic 9 in OT:
throne: 0.0834
hous: 0.0707
lord: 0.0499
twelv: 0.0480
tribe: 0.0410
rejoic: 0.0338
israel: 0.0331
offer: 0.0329
mighti: 0.0312
drink: 0.0281

Top 10 Tokens for Topic 12 in NT:
god: 0.1159
spirit: 0.0651
nation: 0.0440
prophet: 0.0410
lord: 0.0353
judg: 0.0316
jew: 0.0284
kingdom: 0.0279
faith: 0.0269
saint: 0.0265
'''