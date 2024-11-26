import pandas as pd
import numpy as np

# Dictionaries to store system results and relevance judgments
system_results = {}
query_relevance = {}

def load_data():
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

    load_data()
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


if __name__ == '__main__':
    EVAL()
