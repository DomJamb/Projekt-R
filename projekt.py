import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

from util import load_mnist, class_to_onehot
from train_util import get_loss, eval_after_epoch, eval_perf_multi, eval
from test_util import evaluate_model
from attack_funcs import attack_model_fgsm, attack_pgd, attack_model_pgd, train_robust
from graphing_funcs import show_loss, show_train_accuracies, show_weights, graph_stats, graph_details, graph_adv_examples
from AdvExample import AdvExample

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

def train(model, X, Y, param_niter=1000, param_delta=1e-2, param_lambda=1e-3, batch_size=1000, epoch_print=100, conv=False):

    Yoh_ = class_to_onehot(Y.detach().cpu())
    Yoh_ = torch.tensor(Yoh_).to(device)
    
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
            loss = get_loss(probs, y) + (param_lambda * model.get_norm() if not conv else 0)
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

if __name__ == "__main__":
    
    x_train, y_train, x_test, y_test = load_mnist()
    
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    fc_architectures = [[784, 50, 10], [784, 150, 10], [784, 250, 10]]
    no_layers = 3

    # fc_times, fc_accs, conv_times, conv_accs = show_stats(x_train, y_train, x_test, y_test, fc_architectures, no_layers)
    # graph_stats(fc_times, fc_accs, conv_times, conv_accs)
    # graph_details(fc_times, fc_accs, conv_times, conv_accs)

    # model = torch.load('./models/conv_model_2.txt')
    # train_acc, test_acc = evaluate_model(model, x_train, y_train, x_test, y_test, batch_size=500)
    # print("Test accuracy:")
    # print(test_acc)

    # probs = eval(model, x_test.to(device))
    # preds = np.argmax(probs, axis=1)  
    # acc, _ , _ = eval_perf_multi(y_test.detach().cpu().numpy(), preds)
    # print(acc)

    # attack_model_fgsm(model, x_test, y_test)
    # attack_model_pgd(model, x_test, y_test)

    # model = FCmodel(torch.relu, 784, 250, 10).to(device)
    # losses, train_accuracies = train(model, x_train, y_train, param_niter=300, param_delta=0.07, batch_size=50, epoch_print=30)

    # model = ConvModel().to(device)
    # losses, train_accuracies = train(model, x_train, y_train, param_niter=10, param_delta=0.07, batch_size=50, epoch_print=1, conv=True)

    # torch.save(model, './models_old/fcmodel2.txt')

    # show_loss(losses)
    # show_train_accuracies(train_accuracies, 300, "fcmodel1_train_acc.jpg", "./stats/")
    # show_train_accuracies(train_accuracies, 10, "convmodel1_train_acc.jpg", "./stats/")

    ### Training a robust convolutional model

    # conv_model_robust = ConvModel().to(device)
    # losses, train_accuracies = train_robust(conv_model_robust, x_train, y_train, param_niter=10, param_delta=0.07, batch_size=100, epoch_print=1, conv=True)
    # torch.save(conv_model_robust, './models_robust_comparison/robust_model_conv.txt')
    conv_model_robust = torch.load('./models_robust_comparison/robust_model_conv.txt')
    # print(f"Robust model train accuracy: {train_accuracies}")
    
    ### Evaluation of the robust model on the normal dataset

    probs = eval(conv_model_robust, x_test.to(device))
    preds = np.argmax(probs, axis=1)  
    robust_acc, _ , _ = eval_perf_multi(y_test.detach().cpu().numpy(), preds) 
    print(f"Robust model accuracy on normal images: {robust_acc}")

    ### Evaluation of the robust model on adversarial examples

    y_test_oh = class_to_onehot(y_test.detach().cpu())
    y_test_oh = torch.tensor(y_test_oh).to(device)
    adv_images = attack_pgd(conv_model_robust, x_test, y_test_oh, eps=0.3, steps=20)

    adv_probs = eval(conv_model_robust, adv_images.to(device))
    adv_preds = np.argmax(adv_probs, axis=1)  
    robust_adv_acc, _ , _ = eval_perf_multi(y_test.detach().cpu().numpy(), adv_preds) 
    print(f"Robust model accuracy on adversarial images: {robust_adv_acc}")

    ### Generate adversarial examples

    adv_dict = dict()
    for i in range(len(y_test)):
        if y_test[i] in list(adv_dict.keys()):
            continue
        adv_dict.update({y_test[i]: [AdvExample(int(preds[i]), int(adv_preds[i]), x_test[i].detach().cpu().numpy(), adv_images[i].detach().cpu().numpy())]})
        if len(list(adv_dict.keys())) == 3:
            break
    
    graph_adv_examples(adv_dict)


    ### Training a nonrobust convolutional model

    # conv_model = ConvModel().to(device)
    # losses, train_accuracies = train(conv_model, x_train, y_train, param_niter=10, param_delta=0.07, batch_size=100, epoch_print=1, conv=True)
    # torch.save(conv_model, './models_robust_comparison/nonrobust_model_conv.txt')
    conv_model = torch.load('./models_robust_comparison/nonrobust_model_conv.txt')
    # print(f"Nonrobust model train accuracy: {train_accuracies}")

    ### Evaluation of the nonrobust model on the normal dataset

    probs = eval(conv_model, x_test.to(device))
    preds = np.argmax(probs, axis=1)  
    acc, _ , _ = eval_perf_multi(y_test.detach().cpu().numpy(), preds) 
    print(f"Nonrobust model accuracy on normal images: {acc}")

    ### Evaluation of the nonrobust model on adversarial examples

    y_test_oh = class_to_onehot(y_test.detach().cpu())
    y_test_oh = torch.tensor(y_test_oh).to(device)
    adv_images = attack_pgd(conv_model, x_test, y_test_oh, eps=0.3, steps=20)

    adv_probs = eval(conv_model, adv_images.to(device))
    adv_preds = np.argmax(adv_probs, axis=1)  
    adv_acc, _ , _ = eval_perf_multi(y_test.detach().cpu().numpy(), adv_preds) 
    print(f"Nonrobust model accuracy on adversarial images: {adv_acc}")

    ### Generate adversarial examples

    adv_dict = dict()
    for i in range(len(y_test)):
        if y_test[i] in list(adv_dict.keys()):
            continue
        adv_dict.update({y_test[i]: [AdvExample(int(preds[i]), int(adv_preds[i]), x_test[i].detach().cpu().numpy(), adv_images[i].detach().cpu().numpy())]})
        if len(list(adv_dict.keys())) == 3:
            break
    
    graph_adv_examples(adv_dict)

    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))
    
    robust = [robust_acc, robust_adv_acc]
    nonrobust = [acc, adv_acc]
    
    br1 = np.arange(len(robust))
    br2 = [(x + barWidth + 0.05) for x in br1]
    
    plt.bar(br1, robust, color ='g', width = barWidth,
            edgecolor ='grey', label ='Convolutional model with robust training')
    plt.bar(br2, nonrobust, color ='b', width = barWidth,
            edgecolor ='grey', label ='Convolutional model without robust training')
    
    plt.xlabel('Dataset', fontweight ='bold', fontsize = 15)
    plt.ylabel('Accuracy', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(robust))],
            ['Normal images', 'Adversarial images'])
    
    plt.legend()
    # plt.savefig('./stats/robust_nonrobust_acc_comparison.jpg')
    plt.show()