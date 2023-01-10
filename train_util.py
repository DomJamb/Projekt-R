import torch
import numpy as np

def get_loss(X, Yoh_):
    """
    Function for calculating the loss given data and predictions
    Params:
        X: data
        Yoh_: predictions
    Returns:
        loss: calculated loss
    """
    return -torch.mean(torch.sum(Yoh_ * torch.log(X + 1e-20), dim=1))

def eval(model, X):
    """
    Function for calculating the predictions of a model
    Params:
        model: arbitrary deep model which implements the forward function
        X: data
    Returns:
        Y_: predictions for given data X
    """
    return model.forward(X).detach().cpu().numpy()

def eval_after_epoch(model, x, y_):
    """
    Function for calculating the accuracy of a model after a train epoch
    Params:
        model: arbitrary deep model which implements the forward function
        x: data
        y_: correct labels
    Returns:
        acc: accuracy of model on given data 
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 500
    x_batch = torch.split(x, batch_size)

    probs = []
    # Evaluates in batches for better memory management
    for x in x_batch: 
        probs.append(eval(model, x.to(device)))
    probs = np.array(probs).reshape(-1, 10)
    y_pred = np.argmax(probs, axis=1)
    acc, _, _ = eval_perf_multi(y_.numpy(), y_pred)
    return acc

def eval_perf_multi(Y, Y_):
    """
    Function for calculating the accuracy, precision and confusion matrix for given predictions and labels
    Params:
        Y: correct labels
        Y_: predictions
    Returns:
        accuracy: accuracy based on correct labels and prediction
        pr: precision based on correct labels and prediction
        M: confusion matrix based on based on correct labels and prediction
    """
    pr = []
    n = 10
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