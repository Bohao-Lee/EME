import pickle
import numpy as np
import torch
import math
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from torch.autograd import Variable
from classification_net import *
from distribution_calibration import *
from gradient_cutout import * 
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
    n_sample_class = 4
    n_disturbance = 2
    n_iter = 1000
    disturbance_range = 0.6
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples
    
    # net setting
    input_size = 640
    learning_rate = 0.002
    Softmax_fun = torch.nn.Softmax(dim=1)
    loss_func = torch.nn.CrossEntropyLoss()

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
        
        # first training
        for i in range(n_lsamples):
            mean, cov = distribution_calibration_equation_estimation_reweighting(support_data[i], base_means, base_cov, k=n_sample_class)
            
            sampled_data.append(np.random.multivariate_normal(mean=mean, cov=cov, size=num_sampled))
            sampled_label.extend([support_label[i]]*num_sampled)
            
        sampled_data_origin = np.concatenate([sampled_data[:]]).reshape(n_ways * n_shot * num_sampled, -1)
        X_aug = np.concatenate([support_data, sampled_data_origin])
        Y_aug = np.concatenate([support_label, sampled_label])
        # ---- train classifier
        # net classification setting
        classification_net = Net(n_feature=input_size, n_output=n_ways)
        optimizer = torch.optim.SGD(classification_net.parameters(), lr=learning_rate, momentum=0.9)
        
        for i in range(n_iter):
            data_tensor = torch.Tensor(X_aug)
            class_tensor = torch.tensor(Y_aug)
            optimizer.zero_grad()
            out = classification_net(data_tensor)
            loss = loss_func(out, class_tensor)
        
            loss.backward()
            optimizer.step()
        
        # get the grdient map of support data
        for i in range(1):
            data_tensor = Variable(torch.Tensor(support_data), requires_grad=True)
            class_tensor = torch.tensor(support_label)
            # zero the parameter gradients
            optimizer.zero_grad()
            out = classification_net(data_tensor)
            loss = loss_func(out, class_tensor)
        
            loss.backward(retain_graph=True)
            optimizer.step()
        
        
        # novel feature disturbance
        for repeat_time in range(1, n_disturbance):
            
            # gradient cutout
            disturbance_support_data = gradient_cutout(data_tensor.clone(), data_tensor.grad.detach(), ratio=disturbance_range)
            # matrix reweighting disturbance
            for i in range(n_lsamples):
                # matrix reweighting
                mean, cov = distribution_calibration_equation_estimation_disturbance(disturbance_support_data[i].detach(), base_means, base_cov, support_data[i].copy(), k=n_sample_class)
                
                sampled_data.append(np.random.multivariate_normal(mean=mean, cov=cov, size=num_sampled))
                sampled_label.extend([support_label[i]]*num_sampled)
                
            sampled_data_disturbance = np.concatenate([sampled_data[:]]).reshape(n_ways * n_shot * num_sampled * (repeat_time + 1), -1) # 1 is first sample data 
            X_aug = np.concatenate([support_data, sampled_data_disturbance])
            Y_aug = np.concatenate([support_label, sampled_label])
            
        # net classification setting
        classification_net = Net(n_feature=input_size, n_output=n_ways)
        optimizer = torch.optim.SGD(classification_net.parameters(), lr=learning_rate, momentum=0.9)
            
        for i in range(n_iter):
            data_tensor = Variable(torch.Tensor(X_aug))
            class_tensor = torch.tensor(Y_aug)
            # zero the parameter gradients
            optimizer.zero_grad()
            out = classification_net(data_tensor)
            loss = loss_func(out, class_tensor)
        
            loss.backward()
            optimizer.step()
            
        # net predict
        with torch.no_grad():
            # net classification
            query_result = classification_net(torch.Tensor(query_data))
            predicts = torch.max(query_result, 1)[1].cpu().numpy()
        
        acc = np.mean(predicts == query_label)
        acc_list.append(acc)
        
        print('acc %.2f%%, current acc %.2f%%' % (float(np.mean(acc_list)) * 100, acc * 100))
        
        # get final confidence interval
        df = len(acc_list) - 1
        alpha = 0.95
        ci = stats.t.interval(alpha, df, loc=np.mean(acc_list), scale=stats.sem(acc_list))
        print('confidence interval final is (%.2f%%,%.2f%%)' % (ci[0] * 100, ci[1] * 100))
        
    print('%s %d way %d shot  ACC : %f'%(dataset,n_ways,n_shot,float(np.mean(acc_list))))

    # get final confidence interval
    df = len(acc_list) - 1
    alpha = 0.95
    ci = stats.t.interval(alpha, df, loc=np.mean(acc_list), scale=stats.sem(acc_list))
    print('confidence interval final is (%.2f%%,%.2f%%)' % (ci[0] * 100, ci[1] * 100))