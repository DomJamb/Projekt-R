import torch
import torch.nn as nn
import numpy as np
import data
import matplotlib.pyplot as plt
import matplotlib.animation as animation 

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
            w.append(nn.Parameter(torch.randn(neurons[id], neurons[id + 1]), requires_grad=True))
            b.append(nn.Parameter(torch.zeros(neurons[id + 1]), requires_grad=True))        
        self.weigths = nn.ParameterList(w)
        self.biases = nn.ParameterList(b)
        print(self.weigths)
        print(self.biases)

    def count_params(self):
        tensor_shapes = [(p[0], p[1][0].shape) for p in self.named_parameters()]
        total_parameters = np.sum([p.numel() for p in self.parameters()])
        return tensor_shapes, total_parameters

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


def train(model, X, Yoh_, param_niter=400, param_delta=1e-2, param_lambda=1e-3):
    """Arguments:
        - X: model inputs [NxD], type: torch.Tensor
        - Yoh_: ground truth [NxC], type: torch.Tensor
        - param_niter: number of training iterations
        - param_delta: learning rate
    """
    
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

        if epoch % 50 == 0:
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
    return model.forward(torch.Tensor(X)).detach().numpy()

def pt_decfun(model):
    return lambda X: np.argmax(model.forward(torch.tensor(X)).detach().numpy(), axis=1)

if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)

    # instanciraj podatke X i labele Yoh_
    x, Y_ = data.sample_gmm_2d(6, 3, 10)
    X, Yoh_ = torch.tensor(x), torch.tensor(data.class_to_onehot(Y_))

    # definiraj model:
    ptlr = PTDeep(torch.tanh, 2, 10, 10, 3)

    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    train(ptlr, X, Yoh_, param_niter=10000, param_delta=0.1)

    # dohvati vjerojatnosti na skupu za učenje
    probs = eval(ptlr, X)
    Y = np.argmax(probs, axis=1)

    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, pr, M = data.eval_perf_multi(Y, Y_)
    print("Accuracy: {}\nPR:\n{}\nConfusion matrix:\n{}".format(accuracy, pr, M))

    # iscrtaj rezultate, decizijsku plohu
    figure = plt.figure(figsize=(16, 10))
    bbox = (np.min(x, axis=0), np.max(x, axis=0))

    data.graph_surface(pt_decfun(ptlr), bbox, offset=0.5)
    data.graph_data(x, Y_, Y)
    plt.show() 