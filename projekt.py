import torch
import torch.nn as nn
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def class_to_onehot(Y):
    Yoh = np.zeros((len(Y), max(Y) + 1))
    Yoh[range(len(Y)), Y] = 1
    return Yoh

def load_mnist(dataset_root="./data/"):
    mnist_train = datasets.MNIST(dataset_root, train=True, download=False)
    mnist_test = datasets.MNIST(dataset_root, train=False, download=False)

    x_train, y_train = mnist_train.data, mnist_train.targets
    x_test, y_test = mnist_test.data, mnist_test.targets
    x_train, x_test = x_train.float().div_(255.0).reshape(-1, 784), x_test.float().div_(255.0).reshape(-1, 784)

    return x_train, y_train, x_test, y_test

class PTDeep(nn.Module):
    def __init__(self, f, *neurons):
        super().__init__()
        self.f = f
        
        w, b = [], []
        
        for id in range(len(neurons) - 1):
            w.append(nn.Parameter(torch.randn(neurons[id], neurons[id + 1]).to(device), requires_grad=True))
            b.append(nn.Parameter(torch.zeros(neurons[id + 1]).to(device), requires_grad=True))        
        self.weights = nn.ParameterList(w)
        self.biases = nn.ParameterList(b)

    def forward(self, X):
        s = X.float()
        for wi, bi in zip(self.weights[:-1], self.biases[:-1]):
            s = self.f(torch.mm(s, wi) + bi)
        return torch.softmax(torch.mm(s, self.weights[-1]) + self.biases[-1], dim=1)

    def get_loss(self, X, Yoh_):
        return -torch.mean(torch.sum(Yoh_ * torch.log(X + 1e-20), dim=1))

    def get_norm(self):
        norm = 0

        for weights in self.weights:
            norm += torch.norm(weights)

        return norm

#batch_size = 100 postize najbolji rezultat (95.65% acc), ali najduze izvodenje ~ 263s
def train(model, X, Yoh_, param_niter=1000, param_delta=1e-2, param_lambda=1e-3, batch_size=1000):
    if device == "cuda":
        X = X.to(device)
        Yoh_ = Yoh_.to(device)
    
    opt = torch.optim.SGD(model.parameters(), lr=param_delta)
    losses = []

    for epoch in range(param_niter):
        #print(f"_______________Epoha____________ {epoch}")
        permutations = torch.randperm(len(X))
        X_total = X.detach()[permutations]
        Y_total = Yoh_.detach()[permutations]

        X_batch = torch.split(X_total, batch_size)
        Y_batch = torch.split(Y_total, batch_size)

        temp_loss = []

        for i, (x, y) in enumerate(zip(X_batch, Y_batch)):
            #print("Batch = " + str(i))
            probs = model.forward(x)
            loss = model.get_loss(probs, y) + param_lambda * model.get_norm()
            temp_loss.append(loss.detach().cpu().item())
            opt.zero_grad()
            loss.backward()
            opt.step()

        loss = np.mean(temp_loss)
        losses.append(loss)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{param_niter} -> loss = {loss}')
        
    return losses


def eval(model, X):
    return model.forward(torch.Tensor(X)).detach().cpu().numpy()

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

def show_weights(weights):
    fig = plt.figure(figsize=(16, 8))
    # print(len(weights))
    #print(weights[:, 0].detach().cpu().numpy().shape)
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow((weights[:, i].detach().cpu().numpy()).reshape(28, 28))
    plt.show()

def show_loss(loss):
    fig = plt.figure(figsize=(16, 10))
    plt.plot(range(len(loss)), np.array(loss), label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss functions")
    plt.title("Loss function over the epochs")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist()

    # """
    y_train_oh = class_to_onehot(y_train)
    start_time = time.time()
    model = PTDeep(torch.relu, 784, 250, 10).to(device)
    losses = train(model, x_train.cuda(), torch.tensor(y_train_oh).cuda(), param_niter=300, param_delta=0.07, batch_size=50)
    print("--- %s seconds ---" % (time.time() - start_time))
    # """
    # model = torch.load('./fcmodel1.txt')

    model.eval()

    with torch.no_grad():
        probs = eval(model, x_train.cuda())
        y_pred = np.argmax(probs, axis=1)
        acc, pr, m = eval_perf_multi(y_train.numpy(), y_pred)
        print(f"acc = {acc}\npr = {pr}\nm = \n{m}")

        print("-------------------------------------------")

        probs = eval(model, x_test.cuda())
        y_pred = np.argmax(probs, axis=1)
        acc, pr, m = eval_perf_multi(y_test.numpy(), y_pred)
        print(f"acc = {acc}\npr = {pr}\nm = \n{m}")

        #torch.save(model, './model.txt')
        show_loss(losses)
        #print(model.weights[0].detach().cpu().numpy())
        show_weights(model.weights[0])

    model.train()