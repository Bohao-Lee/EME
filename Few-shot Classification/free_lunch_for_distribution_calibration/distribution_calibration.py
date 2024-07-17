import numpy as np
import torch
use_gpu = torch.cuda.is_available()

def softmax(v : [int]):
    l1 = list(map(lambda x: np.exp(x), v))
    return list(map(lambda x: x / sum(l1), l1))

def distribution_calibration(query, base_means, base_cov, k,alpha=0.21):
    dist = []
    for i in range(len(base_means)):
        # get query class to base class distance
        dist.append(np.linalg.norm(query-base_means[i]))
    index = np.argpartition(dist, k)[:k]
    mean = np.concatenate([np.array(base_means)[index], query[np.newaxis, :]])
    calibrated_mean = np.mean(mean, axis=0)
    calibrated_cov = np.mean(np.array(base_cov)[index], axis=0)+alpha

    return calibrated_mean, calibrated_cov


def distribution_calibration_equation_estimation_reweighting(query, base_means, base_cov, k,alpha=0.21):
    dist = []
    near_class_mean = [0.0]
    near_class_cov = [0.0]
    near_base_mean = []
    near_base_cov = []
    
    for i in range(len(base_means)):
        dist.append(np.linalg.norm(query-base_means[i]))
    index = np.argpartition(dist, k)[:k]
    
    for near_index in index:
        near_base_mean.append(base_means[near_index])
        near_base_cov.append(base_cov[near_index])
        
    weight_estimation = np.linalg.pinv(np.array(near_base_mean).T).dot(query)
    # normalized
    weight_estimation = softmax(weight_estimation)
    
    for current_index, current_class_weight in enumerate(weight_estimation):
        near_class_mean[0] = near_class_mean[0] + current_class_weight * near_base_mean[current_index]
        near_class_cov[0] = near_class_cov[0] + abs(current_class_weight) * near_base_cov[current_index]
    mean = np.concatenate([near_class_mean[0][np.newaxis, :], query[np.newaxis, :]])
    calibrated_mean = np.mean(mean, axis=0)
    calibrated_cov = near_class_cov[0] + alpha

    return calibrated_mean, calibrated_cov



def distribution_calibration_equation_estimation_disturbance(query, base_means, base_cov, origin_query, k,alpha=0.21):
    dist = []
    near_class_mean = [0.0]
    near_class_cov = [0.0]
    near_base_mean = []
    near_base_cov = []
    
    for i in range(len(base_means)):
        dist.append(np.linalg.norm(query-base_means[i]))
    index = np.argpartition(dist, k)[:k]
    
    for near_index in index:
        near_base_mean.append(base_means[near_index])
        near_base_cov.append(base_cov[near_index])
        
    weight_estimation = np.linalg.pinv(np.array(near_base_mean).T).dot(query)
    # normalized
    weight_estimation = softmax(weight_estimation)
    
    for current_index, current_class_weight in enumerate(weight_estimation):
        near_class_mean[0] = near_class_mean[0] + current_class_weight * near_base_mean[current_index]
        near_class_cov[0] = near_class_cov[0] + abs(current_class_weight) * near_base_cov[current_index]
    mean = np.concatenate([near_class_mean[0][np.newaxis, :], origin_query[np.newaxis, :]])
    calibrated_mean = np.mean(mean, axis=0)
    calibrated_cov = near_class_cov[0] + alpha

    return calibrated_mean, calibrated_cov

