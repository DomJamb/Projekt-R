import torch
import numpy as np

from train_util import eval, eval_perf_multi

def evaluate_model(model, x_train, y_train, x_test, y_test, batch_size=500):
    """
    Function for calculating the train set accuracy and test set accuracy of a given model
    Params:
        model: arbitrary deep model which implements the forward function
        x_train: train data
        y_train: train labels
        x_test: test data
        y_test: test labels
        batch_size: preferred batch size for testing
    Returns:
        train_acc: train accuracy for  given model
        test_acc: test accuracy for given model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Evaluating the model...")
    model.eval()

    with torch.no_grad():
        X_batch = torch.split(x_train, batch_size)

        #print("----------\nTrain data:\n----------")
        probs = []
        #Evaluates in batches for better memory management
        for x_train in X_batch: 
            probs.append(eval(model, x_train.to(device)))
        probs = np.array(probs).reshape(-1, 10)
        y_pred = np.argmax(probs, axis=1)
        train_acc, pr, m = eval_perf_multi(y_train.detach().cpu().numpy(), y_pred)
        #print(f"Accuracy\n{acc}\nPrecision\n{pr}\nConfusion matrix\n{m}")

        #print("----------\nTest data:\n----------")
        
        probs = eval(model, x_test.to(device))
        y_pred = np.argmax(probs, axis=1)
        test_acc, pr, m = eval_perf_multi(y_test.detach().cpu().numpy(), y_pred)
        #print(f"Accuracy\n{acc}\nPrecision\n{pr}\nConfusion matrix\n{m}")
        
    model.train()
    print("Finished evaluating the model...")

    return train_acc, test_acc