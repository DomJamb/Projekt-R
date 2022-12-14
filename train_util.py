import torch
import numpy as np

def get_loss(X, Yoh_):
    return -torch.mean(torch.sum(Yoh_ * torch.log(X + 1e-20), dim=1))

def eval(model, X):
    return model.forward(torch.tensor(X)).detach().cpu().numpy()

def eval_after_epoch(model, x, y_):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 500
    x_batch = torch.split(x, batch_size)

    probs = []
    for x in x_batch: 
        probs.append(eval(model, x.to(device)))
    probs = np.array(probs).reshape(-1, 10)
    y_pred = np.argmax(probs, axis=1)
    acc, _, _ = eval_perf_multi(y_.numpy(), y_pred)
    return acc

def eval_perf_multi(Y, Y_):
    pr = []
    n = max(Y_) + 1
    M = np.bincount(n * Y_ + Y, minlength=n * n).reshape(n, n)
    for i in range(n):
        tp_i = M[i, i]
        fn_i = np.sum(M[i, :]) - tp_i
        fp_i = np.sum(M[:, i]) - tp_i
        tn_i = np.sum(M) - fp_i - fn_i - tp_i
        recall_i = tp_i / (tp_i + fn_i)
        precision_i = tp_i / (tp_i + fp_i)
        pr.append((recall_i, precision_i))

    accuracy = np.trace(M) / np.sum(M)

    return accuracy, pr, M