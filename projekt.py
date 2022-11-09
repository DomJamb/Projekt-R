import torch
import torch.nn as nn
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt

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
        """Arguments:
        - D: dimensions of each datapoint
        """
        # inicijalizirati parametre (koristite nn.Parameter):
        # imena mogu biti self.W, self.b
        super().__init__()
        self.f = f
        
        w, b = [], []
        
        for id in range(len(neurons) - 1):
            w.append(nn.Parameter(torch.randn(neurons[id], neurons[id + 1]).to(device), requires_grad=True))
            b.append(nn.Parameter(torch.zeros(neurons[id + 1]).to(device), requires_grad=True))        
        self.weigths = nn.ParameterList(w)
        self.biases = nn.ParameterList(b)
        #print(self.weigths)
        #print(self.biases)

    # def count_params(self):
    #     tensor_shapes = [(p[0], p[1][0].shape) for p in self.named_parameters()]
    #     total_parameters = np.sum([p.numel() for p in self.parameters()])
    #     return tensor_shapes, total_parameters

    def forward(self, X):
        # unaprijedni prolaz modela: izračunati vjerojatnosti
        #   koristiti: torch.mm, torch.softmax
        s = X.float()
        for wi, bi in zip(self.weigths[:-1], self.biases[:-1]):
            s = self.f(torch.mm(s, wi) + bi)
        return torch.softmax(torch.mm(s, self.weigths[-1]) + self.biases[-1], dim=1)

    def get_loss(self, X, Yoh_):
        # formulacija gubitka
        #   koristiti: torch.log, torch.mean, torch.sum
        return -torch.mean(torch.sum(Yoh_ * torch.log(X + 1e-20), dim=1))


def train(model, X, Yoh_, param_niter=1000, param_delta=1e-2, param_lambda=1e-3):
    """Arguments:
        - X: model inputs [NxD], type: torch.Tensor
        - Yoh_: ground truth [NxC], type: torch.Tensor
        - param_niter: number of training iterations
        - param_delta: learning rate
    """

    if device == "cuda":
        X = X.to(device)
        Yoh_ = Yoh_.to(device)
    
    # inicijalizacija optimizatora
    opt = torch.optim.SGD(model.parameters(), lr=param_delta)

    # petlja učenja
    # ispisujte gubitak tijekom učenja
    for epoch in range(param_niter):
        probs = model.forward(X)
        loss = model.get_loss(probs, Yoh_)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{param_niter} -> loss = {loss}')
    return


def eval(model, X):
    """Arguments:
        - model: type: PTLogreg
        - X: actual datapoints [NxD], type: np.array
        Returns: predicted class probabilites [NxC], type: np.array
    """
    # ulaz je potrebno pretvoriti u torch.Tensor
    # izlaze je potrebno pretvoriti u numpy.array
    # koristite torch.Tensor.detach() i torch.Tensor.numpy()
    return model.forward(torch.Tensor(X)).detach().cpu().numpy()

def eval_after_epoch(self, x_val, y_val):
    y_pred = self.ev(x_val)

    y_true = np.argmax(y_val, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    cm_diag = np.diag(cm)

    sums = [np.sum(cm, axis=y) for y in [None, 0, 1]]

    sums[0] = np.maximum(1, sums[0])
    for i in range(1, len(sums)):
        sums[i][sums[i] == 0] = 1

    accuracy = np.sum(cm_diag) / sums[0]
    precision, recall = [np.mean(cm_diag / x) for x in sums[1:]]
    f1 = (2 * precision * recall) / (precision + recall)

    return {"acc": accuracy, "pr": precision, "re": recall, "f1": f1}


def pt_decfun(model):
    return lambda X: np.argmax(model.forward(torch.tensor(X)).detach().numpy(), axis=1)

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

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist()
    # print(x_test.shape)
    # print(y_train.shape)
    # first_image = x_test[0]
    # first_image = np.array(first_image, dtype='float')
    # pixels = first_image.reshape((28, 28))
    # plt.imshow(pixels, cmap='gray')
    # plt.show()

    y_train_oh = class_to_onehot(y_train)
    model = PTDeep(torch.relu, 784, 100, 10).to(device)
    train(model, x_train.cuda(), torch.tensor(y_train_oh).cuda(), param_niter=1000, param_delta=0.1)

    probs = eval(model, x_test.cuda())
    y_pred = np.argmax(probs, axis=1)

    print(type(y_test), type(y_pred))
    acc, pr, m = eval_perf_multi(y_test.numpy(), y_pred)
    print(f"acc = {acc}\npr = {pr}\nm = \n{m}")