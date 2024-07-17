import pickle
import numpy as np
import torch
import math
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from classification_net import *
from distribution_calibration import *
from scipy import stats
use_gpu = torch.cuda.is_available()

if __name__ == '__main__':
    # ---- data loading
    dataset = 'miniImagenet'
    #dataset = 'CUB'
    n_shot = 1
    n_ways = 5
    n_queries = 15
    n_runs = 600
    n_sample_class = 6
    n_iter = 1000
    disturance_quantity = 0.001
    disturbance_range = 0.15
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples

    import FSLTask
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
    FSLTask.loadDataSet(dataset)
    FSLTask.setRandomStates(cfg)
    ndatas = FSLTask.GenerateRunSet(end=n_runs, cfg=cfg)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 5).clone().view(n_runs,
                                                                                                        n_samples)
    # ---- Base class statistics
    base_means = []
    base_cov = []
    base_features_path = "./checkpoints/%s/WideResNet28_10_S2M2_R/last/base_features.plk"%dataset
    
    with open(base_features_path, 'rb') as f:
        data = pickle.load(f)
        for key in data.keys():
            feature = np.array(data[key])
            mean = np.mean(feature, axis=0)
            cov = np.cov(feature.T)
            base_means.append(mean)
            base_cov.append(cov)

    # ---- classification for each task
    acc_list = []
    print('Start classification for %d tasks...'%(n_runs))
    for i in tqdm(range(n_runs)):

        support_data = ndatas[i][:n_lsamples].numpy()
        support_label = labels[i][:n_lsamples].numpy()
        query_data = ndatas[i][n_lsamples:].numpy()
        query_label = labels[i][n_lsamples:].numpy()
        # ---- Tukey's transform
        beta = 0.5
        support_data = np.power(support_data[:, ] ,beta)
        query_data = np.power(query_data[:, ] ,beta)
        # ---- distribution calibration and feature sampling
        sampled_data = []
        sampled_label = []
        num_sampled = int(750/n_shot)
        for i in range(n_lsamples):
            # baseline
            #mean, cov = distribution_calibration(support_data[i], base_means, base_cov, k=n_sample_class)
            # matrix sparse reweighting
            mean, cov = distribution_calibration_equation_estimation_reweighting(support_data[i], base_means, base_cov, k=n_sample_class)
            
            sampled_data.append(np.random.multivariate_normal(mean=mean, cov=cov, size=num_sampled))
            sampled_label.extend([support_label[i]]*num_sampled)
            
        sampled_data = np.concatenate([sampled_data[:]]).reshape(n_ways * n_shot * num_sampled, -1)
        X_aug = np.concatenate([support_data, sampled_data])
        Y_aug = np.concatenate([support_label, sampled_label])
        # ---- train classifier
        classifier = LogisticRegression(max_iter=1000).fit(X=X_aug, y=Y_aug)

        # logistic regression
        predicts = classifier.predict(query_data)
        acc = np.mean(predicts == query_label)
        acc_list.append(acc)
        
        print('acc %.2f%%, current acc %.2f%%' % (float(np.mean(acc_list)) * 100, acc * 100))
        df = len(acc_list) - 1
        alpha = 0.95
        ci = stats.t.interval(alpha, df, loc=np.mean(acc_list), scale=stats.sem(acc_list))
        print('confidence interval final is (%.2f%%,%.2f%%)' % (ci[0] * 100, ci[1] * 100))
        
    print('%s %d way %d shot  ACC : %f'%(dataset,n_ways,n_shot,float(np.mean(acc_list))))
    df = len(acc_list) - 1
    alpha = 0.95
    ci = stats.t.interval(alpha, df, loc=np.mean(acc_list), scale=stats.sem(acc_list))
    print('confidence interval final is (%.2f%%,%.2f%%)' % (ci[0] * 100, ci[1] * 100))
