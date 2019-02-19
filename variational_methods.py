
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch import FloatTensor as FT

def log_rect(x):
    x = t.where(x > 0,t.log(1+f.relu(x)),x)
    return x

def gauss_log_p(param,x):
    mean,log_var = param
    log_p = -((x-mean)**2)/(2*t.exp(log_var))-log_var/2 - math.log(2*math.pi)/2
    return log_p

def gauss_samp(param):
    mean,log_var = param
    return t.randn_like(mean)*t.exp(log_var/2)+mean

def gauss_ent(param):
    mean,log_var = param
    return log_var/2 - math.log(2*math.pi*math.e)/2

def mean_sum(x):
    return t.mean(t.einsum('i...->i',(x,)))

def gauss_kl_div(p,q):
    p_mean,p_log_var = p
    q_mean,q_log_var = q
    p_log_std = p_log_var/2
    q_log_std = q_log_var/2
    p_var = t.exp(p_log_var)
    q_var = t.exp(q_log_var)
    kl_div = q_log_std-p_log_std +(p_var + (p_mean-q_mean)**2)/(2*q_var)-1/2
    return(kl_div)
