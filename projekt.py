import torch
import torch.nn as nn
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FCmodel(nn.Module):
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

class ConvModel(nn.Module):
    def __init__(self, no_layers=2):
        super().__init__()

        conv = list()
        maxpool = list()
        fc = list()

        in_channels = 1
        out_channels = 16
        input_dim = 28

        weights = list()
        biases = list()

        for i in range(no_layers):
            conv.append(nn.Conv2d(in_channels, out_channels, kernel_size=5, padding="same"))
            in_channels = out_channels
            out_channels *= 2
            maxpool.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
            input_dim = round(input_dim/2)
            total = input_dim**2 * in_channels
            weights.append(nn.init.kaiming_normal_(conv[-1].weight))      
            biases.append(nn.init.constant_(conv[-1].bias, 0.))

        if (total > 2048):
            fc.append(nn.Linear(in_features=total, out_features=1024))
            fc.append(nn.Linear(in_features=1024, out_features=512))
            fc.append(nn.Linear(in_features=512, out_features=10))
        else:
            fc.append(nn.Linear(in_features=total, out_features=512))
            fc.append(nn.Linear(in_features=512, out_features=10))

        for fc_layer in fc:
            weights.append(nn.init.kaiming_normal_(fc_layer.weight))
            biases.append(nn.init.constant_(fc_layer.bias, 0.))

        self.conv = conv
        self.maxpool = maxpool
        self.fc = fc
        self.weights = nn.ParameterList(weights)
        self.biases = nn.ParameterList(biases)

    def forward(self, X):
        
        X = X.reshape(-1, 1, 28, 28).float()
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)

        X.to(device)

        conv = X
        for i in range(len(self.conv)):
            conv = self.conv[i](conv)
            conv = self.maxpool[i](conv)
            conv = torch.relu(conv)

        fc = conv.view((conv.shape[0], -1))

        for i in range(len(self.fc) - 1):
            fc = self.fc[i](fc)
            fc = torch.relu(fc)

        fc = self.fc[-1](fc)
        softmax = torch.softmax(fc, dim=1)
        return softmax

    def get_loss(self, X, Yoh_):
        return -torch.mean(torch.sum(Yoh_ * torch.log(X + 1e-20), dim=1))

def load_mnist(dataset_root="./data/"):
    mnist_train = datasets.MNIST(dataset_root, train=True, download=False)
    mnist_test = datasets.MNIST(dataset_root, train=False, download=False)

    x_train, y_train = mnist_train.data, mnist_train.targets
    x_test, y_test = mnist_test.data, mnist_test.targets
    x_train, x_test = x_train.float().div_(255.0).reshape(-1, 784), x_test.float().div_(255.0).reshape(-1, 784)

    return x_train, y_train, x_test, y_test

def class_to_onehot(Y):
    Yoh = np.zeros((len(Y), max(Y) + 1))
    Yoh[range(len(Y)), Y] = 1
    return Yoh

def train(model, X, Y, param_niter=1000, param_delta=1e-2, param_lambda=1e-3, batch_size=1000, epoch_print=100, conv=False):

    Yoh_ = class_to_onehot(Y.detach().cpu())
    Yoh_ = torch.tensor(Yoh_).cuda()
    
    opt = torch.optim.SGD(model.parameters(), lr=param_delta)
    losses = []
    train_accuracies = []

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
            loss = model.get_loss(probs, y) + (param_lambda * model.get_norm() if not conv else 0)
            temp_loss.append(loss.detach().cpu().item())
            opt.zero_grad()
            loss.backward()
            opt.step()

        loss = np.mean(temp_loss)
        losses.append(loss)

        if epoch % epoch_print == 0:
            print(f'Epoch {epoch}/{param_niter} -> loss = {loss}')
            train_accuracies.append(eval_after_epoch(model, X, Y.detach().cpu()))
        
    return losses, train_accuracies


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

def eval_after_epoch(model, x, y_):
    batch_size = 500
    x_batch = torch.split(x, batch_size)

    probs = []
    for x in x_batch: 
        probs.append(eval(model, x.cuda()))
    probs = np.array(probs).reshape(-1, 10)
    y_pred = np.argmax(probs, axis=1)
    acc, _, _ = eval_perf_multi(y_.numpy(), y_pred)
    return acc

def show_weights(weights):
    fig = plt.figure(figsize=(16, 8))
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

def show_train_accuracies(accs, epochs, name, path):
    fig = plt.figure(figsize=(16,5))
    epochs_step = epochs / 10
    epochs = np.arange(0, epochs, epochs_step)
    plt.plot(epochs, np.array(accs))
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Train accuracy over the epochs")
    plt.savefig(path + name)
    plt.show()

def evaluate_model(model, x_train, y_train, x_test, y_test, batch_size=500):
    print("Evaluating the model...")
    model.eval()

    with torch.no_grad():
        
        X_batch = torch.split(x_train, batch_size)

        #print("----------\nTrain data:\n----------")
        probs = []
        for x_train in X_batch: 
            probs.append(eval(model, x_train.cuda()))
        probs = np.array(probs).reshape(-1, 10)
        y_pred = np.argmax(probs, axis=1)
        train_acc, pr, m = eval_perf_multi(y_train.detach().cpu().numpy(), y_pred)
        #print(f"Accuracy\n{acc}\nPrecision\n{pr}\nConfusion matrix\n{m}")

        #print("----------\nTest data:\n----------")
        
        probs = eval(model, x_test.cuda())
        y_pred = np.argmax(probs, axis=1)
        test_acc, pr, m = eval_perf_multi(y_test.detach().cpu().numpy(), y_pred)
        #print(f"Accuracy\n{acc}\nPrecision\n{pr}\nConfusion matrix\n{m}")
        
    model.train()
    print("Finished evaluating the model...")

    return train_acc, test_acc

def show_stats(x_train, y_train, x_test, y_test, fc_architectures, no_layers):
    fc_accs = list()
    fc_times = list()

    conv_accs = list()
    conv_times = list()
    i = 1
    
    for architecture in fc_architectures:
        start_time = time.time()
        fc_model = FCmodel(torch.relu, *architecture).to(device)
        losses, train_accuracies = train(fc_model, x_train, y_train, param_niter=300, param_delta=0.07, batch_size=200, epoch_print=30) #optimalno batch_size = 50

        fc_times.append(time.time() - start_time)
        #torch.save(fc_model, f'./models/fc_model_{i}.txt')
        i += 1
        train_acc, test_acc = evaluate_model(fc_model, x_train, y_train, x_test, y_test)
        fc_accs.append(test_acc)

    for i in range(no_layers):
        start_time = time.time()
        conv_model = ConvModel(i+1).to(device)
        losses, train_accuracies = train(conv_model, x_train, y_train, param_niter=10, param_delta=0.07, batch_size=200, epoch_print=1, conv=True)

        conv_times.append(time.time() - start_time)
        #torch.save(conv_model, f'./models/conv_model_{i+1}.txt')
        train_acc, test_acc = evaluate_model(conv_model, x_train, y_train, x_test, y_test)
        conv_accs.append(test_acc)
    
    print("FC model times: ")
    print(fc_times)
    print("FC model accuracies: ")
    print(fc_accs)

    print("Conv model times: ")
    print(conv_times)
    print("Conv model accuracies: ")
    print(conv_accs)

    return fc_times, fc_accs, conv_times, conv_accs

def graph_stats(fc_times, fc_accs, conv_times, conv_accs):
    fig = plt.figure(figsize=(16,5))

    plt.plot(fc_accs, fc_times, 'r', label="FC model")
    plt.plot(conv_accs, conv_times, 'b', label="Conv model")
    plt.xlabel("Accuracies")
    plt.ylabel("Train times")
    plt.title("Accuracy/train time graph for FC and Conv models")
    plt.legend()
    plt.show()

def graph_details(fc_times, fc_accs, conv_times, conv_accs):

    layer_num = [1, 2, 3]
    fig = plt.figure(figsize=(16,10))

    plt.subplot(2, 1, 1)
    plt.plot(layer_num, fc_times, 'r', label="FC model")
    plt.plot(layer_num, conv_times, 'b', label="Conv model")
    plt.xlabel("Number of layers")
    plt.ylabel("Train time")
    plt.title("Number of layers / Train time graph for FC and Conv models")
    plt.legend(loc="center right")

    plt.subplot(2, 1, 2)
    plt.plot(layer_num, fc_accs, 'r', label="FC model")
    plt.plot(layer_num, conv_accs, 'b', label="Conv model")
    plt.xlabel("Number of layers")
    plt.ylabel("Accuracy")
    plt.title("Number of layers / Accuracy graph for FC and Conv models")
    plt.legend(loc="center right")
    #plt.savefig('./stats/statistics.jpg')
    plt.show()

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist()
    
    x_train = x_train.cuda()
    y_train = y_train.cuda()

    fc_architectures = [[784, 50, 10], [784, 150, 10], [784, 250, 10]]
    no_layers = 3

    #fc_times, fc_accs, conv_times, conv_accs = show_stats(x_train, y_train, x_test, y_test, fc_architectures, no_layers)
    #graph_stats(fc_times, fc_accs, conv_times, conv_accs)
    #graph_details(fc_times, fc_accs, conv_times, conv_accs)

    model = torch.load('./models/conv_model_2.txt')
    train_acc, test_acc = evaluate_model(model, x_train, y_train, x_test, y_test, batch_size=500)
    print("Test accuracy:")
    print(test_acc)

    """
    start_time = time.time()
    model = FCmodel(torch.relu, 784, 250, 10).to(device)
    losses, train_accuracies = train(model, x_train, y_train, param_niter=300, param_delta=0.07, batch_size=50, epoch_print=30)

    # model = ConvModel().to(device)
    # losses, train_accuracies = train(model, x_train, y_train, param_niter=10, param_delta=0.07, batch_size=50, epoch_print=1, conv=True)

    print("--- %s seconds ---" % (time.time() - start_time))
    #torch.save(model, './models/fcmodel2.txt')

    model = torch.load('./models/convmodel2.txt')

    show_loss(losses)
    show_weights(model.weights[0])
    show_train_accuracies(train_accuracies, 300, "fcmodel1_train_acc.jpg", "./stats/")
    show_train_accuracies(train_accuracies, 10, "convmodel1_train_acc.jpg", "./stats/")
    """    