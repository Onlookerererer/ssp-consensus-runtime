import numpy as np
import scipy.spatial.distance


def fx_calc_map_label(image_features, text_features, labels, k=0, dist_method='COS'):
    if dist_method == 'L2':
        dist_matrix = scipy.spatial.distance.cdist(image_features, text_features, 'euclidean')
    elif dist_method == 'COS':
        dist_matrix = scipy.spatial.distance.cdist(image_features, text_features, 'cosine')
    else:
        raise ValueError(f"Unsupport: {dist_method}. Please use 'L2' or 'COS'. ")

    sorted_indices = dist_matrix.argsort()
    num_queries = dist_matrix.shape[0]
    if k == 0:
        k = num_queries

    average_precisions = []
    for i in range(num_queries):
        ranked_indices = sorted_indices[i]
        true_label = labels[i]

        precision_sum = 0.0
        relevant_count = 0
        for j in range(k):
            retrieved_index = ranked_indices[j]
            if labels[retrieved_index] == true_label:
                relevant_count += 1
                precision_sum += relevant_count / (j + 1)
        if relevant_count > 0:
            ap = precision_sum / relevant_count
            average_precisions.append(ap)
        else:
            average_precisions.append(0.0)

    mean_ap = np.mean(average_precisions)
    return mean_ap


def fx_calc_map_multilabel(query_features, database_features, query_labels, k=0, metric='cosine'):
    dist_matrix = scipy.spatial.distance.cdist(query_features, database_features, metric)
    sorted_indices = dist_matrix.argsort()

    num_queries = dist_matrix.shape[0]
    if k == 0:
        k = dist_matrix.shape[1]

    average_precisions = []
    for i in range(num_queries):
        ranked_indices = sorted_indices[i]
        true_label_vec = query_labels[i]
        top_k_ranked_indices = ranked_indices[:k]
        is_relevant = (np.dot(query_labels[top_k_ranked_indices], true_label_vec) > 0)
        if not np.any(is_relevant):
            average_precisions.append(0.0)
            continue
        cumulative_relevant = np.cumsum(is_relevant)
        precision_at_j = cumulative_relevant / np.arange(1.0, len(is_relevant) + 1)
        precision_at_relevant = precision_at_j[is_relevant]

        ap = np.mean(precision_at_relevant)
        average_precisions.append(ap)

    mean_ap = np.mean(average_precisions)
    return mean_ap