import torch
import numpy as np

def random_earse(input_prototype, ratio=0.15):
    prototype_num = input_prototype.shape[0]
    output_prototype = input_prototype.clone()
    for current_index in range(prototype_num):
        current_prototype = output_prototype[current_index]
        indices = np.random.choice(np.arange(current_prototype.shape[0]), replace=False, size=int(current_prototype.shape[0] * ratio))
        output_prototype[current_index][indices] = 0
    
    return output_prototype

def gradient_cutout(input_prototype, gradient, ratio=0.15):
    prototype_num = input_prototype.shape[0]
    output_prototype = input_prototype.clone()
    for current_index in range(prototype_num):
        current_prototype = output_prototype[current_index]
        current_gradient = gradient[current_index]
        current_sort_gradient = np.sort(current_gradient)
        if ratio > 0:
            cutout_value = current_sort_gradient[int(current_gradient.shape[0] * (1 - ratio))]
            output_prototype[current_index][current_gradient >= cutout_value] = 0
        else:
            continue
    
    return output_prototype

def gradient_cutout_setting(input_prototype, gradient, upper_bound=0.00, lower_bound=0.15):
    prototype_num = input_prototype.shape[0]
    output_prototype = input_prototype.clone()
    for current_index in range(prototype_num):
        current_prototype = output_prototype[current_index]
        current_gradient = gradient[current_index]
        current_sort_gradient = np.sort(current_gradient)
        
        upper_value = current_sort_gradient[int(current_gradient.shape[0] * (1 - upper_bound) - 1)]
        lower_value = current_sort_gradient[int(current_gradient.shape[0] * (1 - lower_bound))]
        
        upper_mask = current_prototype >= lower_value
        lower_mask = current_prototype <= upper_value
        mask = upper_mask * lower_mask
        
        output_prototype[current_index] = current_prototype * (~mask).float()
    
    return output_prototype
