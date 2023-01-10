import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

from util import load_mnist, class_to_onehot
from train_util import get_loss, eval_after_epoch, eval_perf_multi, eval
from test_util import evaluate_model
from attack_funcs import attack_model_fgsm, attack_pgd, attack_model_pgd, train_robust, attack_pgd_directed
from graphing_funcs import show_loss, show_train_accuracies, show_weights, graph_stats, graph_details, graph_adv_examples, graph_targeted_examples
from AdvExample import AdvExample
from TargetedAdvExample import TargetedAdvExample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FCmodel(nn.Module):
    """
    Fully connected model implementation with custom architecture
    """
    def __init__(self, f, *neurons):
        """
        Constructor for the fully connected model
        Params:
            f: nonlinear function
            neurons: list of neurons for each layer (e.g. [784, 250, 10] - 784 neurons in first layer, 10 neurons in final layer (output), 250 neurons in hidden layer)
        """
        super().__init__()
        self.f = f
        
        w, b = [], []
        
        for id in range(len(neurons) - 1):
            # Initialization of weights using normal distribution
            w.append(nn.Parameter(torch.randn(neurons[id], neurons[id + 1]).to(device), requires_grad=True))
            # Initialization of bias to zeros
            b.append(nn.Parameter(torch.zeros(neurons[id + 1]).to(device), requires_grad=True))        
        self.weights = nn.ParameterList(w)
        self.biases = nn.ParameterList(b)

    def forward(self, X):
        """
        Forward pass for the fully connected model
        Params: 
            X: data
        Returns:
            probs: softmaxed output of net
        """
        s = X.float()
        for wi, bi in zip(self.weights[:-1], self.biases[:-1]):
            s = self.f(torch.mm(s, wi) + bi)
        return torch.softmax(torch.mm(s, self.weights[-1]) + self.biases[-1], dim=1)

    def get_norm(self):
        """
        Function for calculating the norm of weights in each layer
        Returns:
            norm: calculated norm of weights
        """
        norm = 0

        for weights in self.weights:
            norm += torch.norm(weights)

        return norm

class ConvModel(nn.Module):
    """
    Convolutional model with arbitrary number of layers
    """
    def __init__(self, no_layers=2):
        """
        Constructor for convolutional model
        Params:
            no_layers: number of layers for the given model
        """
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
            # Convolutional layer with 5x5 kernel and padding
            conv.append(nn.Conv2d(in_channels, out_channels, kernel_size=5, padding="same"))
            in_channels = out_channels
            out_channels *= 2
            # 2x2 Maxpool with stride=2 and ceil for uneven dimensions (e.g. maxpooling 7x7)
            maxpool.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
            input_dim = round(input_dim/2)
            total = input_dim**2 * in_channels
            # Initialization of weights using kaiming normal distribution
            weights.append(nn.init.kaiming_normal_(conv[-1].weight))    
            # Initialization of biases using zeros  
            biases.append(nn.init.constant_(conv[-1].bias, 0.))

        # Number of fully connected layers depends on the number of input features
        if (total > 2048):
            fc.append(nn.Linear(in_features=total, out_features=1024))
            fc.append(nn.Linear(in_features=1024, out_features=512))
            fc.append(nn.Linear(in_features=512, out_features=10))
        else:
            fc.append(nn.Linear(in_features=total, out_features=512))
            fc.append(nn.Linear(in_features=512, out_features=10))

        for fc_layer in fc:
            # Initialization of weights using kaiming normal distribution
            weights.append(nn.init.kaiming_normal_(fc_layer.weight))
            # Initialization of biases using zeros  
            biases.append(nn.init.constant_(fc_layer.bias, 0.))

        self.conv = conv
        self.maxpool = maxpool
        self.fc = fc
        self.weights = nn.ParameterList(weights)
        self.biases = nn.ParameterList(biases)

    def forward(self, X):
        """
        Forward pass for the convolutional model
        Params: 
            X: data
        Returns:
            probs: softmaxed output of net
        """
        X = X.reshape(-1, 1, 28, 28).float()
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)

        X.to(device)

        conv = X
        # Iterate over every convolutional layer to get output for fully connected layers
        for i in range(len(self.conv)):
            conv = self.conv[i](conv)
            conv = self.maxpool[i](conv)
            conv = torch.relu(conv)

        # Flatten
        fc = conv.view((conv.shape[0], -1))

        # Iterate over every fully connected layer except the last 
        for i in range(len(self.fc) - 1):
            fc = self.fc[i](fc)
            fc = torch.relu(fc)

        # Softmax for final FC layer
        fc = self.fc[-1](fc)
        softmax = torch.softmax(fc, dim=1)
        return softmax

def train(model, X, Y, param_niter=1000, param_delta=1e-2, param_lambda=1e-3, batch_size=1000, epoch_print=100, conv=False):
    """
    Function for training a given model on given data
    Params: 
        X: data
        Y: correct labels
        param_niter: number of epochs
        param_delta: learning rate
        param_lambda: coefficient for regularization
        batch_size: arbitrary batch size
        epoch_print: number which determines how often the results of training are printed
        conv: boolean, True signifies that the model is a convolutional model
    Returns:
        losses: calculated losses for each epoch
        train_accuracies: calculated train set accuracies for each epoch
    """
    Yoh_ = class_to_onehot(Y.detach().cpu())
    Yoh_ = torch.tensor(Yoh_).to(device)
    
    # Using SGD optimizer with param_delta learning rate
    opt = torch.optim.SGD(model.parameters(), lr=param_delta)
    losses = []
    train_accuracies = []

    for epoch in range(param_niter):
        #print(f"_______________Epoha____________ {epoch}")
        # Shuffle the data and split it into batches
        permutations = torch.randperm(len(X))
        X_total = X.detach()[permutations]
        Y_total = Yoh_.detach()[permutations]

        X_batch = torch.split(X_total, batch_size)
        Y_batch = torch.split(Y_total, batch_size)

        temp_loss = []

        # Iterate over batches in epoch and do the optimization step
        for i, (x, y) in enumerate(zip(X_batch, Y_batch)):
            #print("Batch = " + str(i))
            probs = model.forward(x)
            loss = get_loss(probs, y) + (param_lambda * model.get_norm() if not conv else 0)
            temp_loss.append(loss.detach().cpu().item())
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Calculate mean loss over each batch in epoch
        loss = np.mean(temp_loss)
        losses.append(loss)

        if epoch % epoch_print == 0:
            print(f'Epoch {epoch}/{param_niter} -> loss = {loss}')
            train_accuracies.append(eval_after_epoch(model, X, Y.detach().cpu()))
        
    return losses, train_accuracies

def show_stats(x_train, y_train, x_test, y_test, fc_architectures, no_layers):
    """
    Function for measuring train time and test set accuracies for fully connected and convolutional models
    Params:
        x_train: train data
        y_train: train labels
        x_test: test data
        y_test: test labels
        fc_architectures: list of lists signifying arbitrary fully connected architectures (e.g. [[784, 100, 10], [784, 250, 10]])
        no_layers: list of desired numbers of layers for each convolutional model (e.g. [2, 3] - first model has 2 convolutional layers, the second model has 3)
    Returns:
        fc_times: measured train times for fully connected models
        fc_accs: measured accuracies on test data for fully connected models
        conv_times: measured train times for convolutional models
        conv_accs: measured accuracies on test data for convolutional models
    """
    fc_accs = list()
    fc_times = list()

    conv_accs = list()
    conv_times = list()
    i = 1
    
    # Train each fully connected model and evaluate it
    for architecture in fc_architectures:
        start_time = time.time()
        fc_model = FCmodel(torch.relu, *architecture).to(device)
        losses, train_accuracies = train(fc_model, x_train, y_train, param_niter=300, param_delta=0.07, batch_size=200, epoch_print=30) #optimalno batch_size = 50

        fc_times.append(time.time() - start_time)
        #torch.save(fc_model, f'./models/fc_model_{i}.txt')
        i += 1
        train_acc, test_acc = evaluate_model(fc_model, x_train, y_train, x_test, y_test)
        fc_accs.append(test_acc)

    # Train each convolutional model and evaluate it
    for i in range(no_layers):
        start_time = time.time()
        conv_model = ConvModel(i+1).to(device)
        losses, train_accuracies = train(conv_model, x_train, y_train, param_niter=10, param_delta=0.07, batch_size=200, epoch_print=1, conv=True)

        conv_times.append(time.time() - start_time)
        #torch.save(conv_model, f'./models/conv_model_{i+1}.txt')
        train_acc, test_acc = evaluate_model(conv_model, x_train, y_train, x_test, y_test)
        conv_accs.append(test_acc)
    
    #Print evaluation results for all models
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
    # Load mnist dataset
    x_train, y_train, x_test, y_test = load_mnist()
    
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    # Arbitrary fully connected model architectures and number of layers for convolutional model
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
    
    ### Generation of targeted attack images and evaluation on them
    
    adv_dict = dict()
    no_of_steps = 100
    epsilon = 0.75
    koef_iteration = 0.035
    for desired_class in range(5):
        print(f"Generating adverserial examples with target class {desired_class}...")
        targeted_adv_images = attack_pgd_directed(conv_model_robust, x_test, y_test, target_class=desired_class, steps=no_of_steps, eps=epsilon, koef_it=koef_iteration)
        probs = eval(conv_model_robust, targeted_adv_images.to(device))
        preds = np.argmax(probs, axis=1)
        targeted_acc, _ , _ = eval_perf_multi(y_test.detach().cpu().numpy(), preds) 
        print(f"Robust model accuracy on targeted PGD images with target class {desired_class}: {targeted_acc}")

        targeted_examples = targeted_adv_images.detach().cpu().numpy()
        adv_list = list()
        for i in range(4):
            adv_list.append(TargetedAdvExample(x_test[i], targeted_examples[i]))
        adv_dict.update({desired_class: adv_list})

    graph_targeted_examples(adv_dict, pathname=f"./stats/targeted_adv_eps_{epsilon}_steps_{no_of_steps}_it_{koef_iteration}_1.jpg")

    adv_dict = dict()
    for desired_class in range(5,10):
        print(f"Generating adverserial examples with target class {desired_class}...")
        targeted_adv_images = attack_pgd_directed(conv_model_robust, x_test, y_test, target_class=desired_class, steps=no_of_steps, eps=epsilon, koef_it=koef_iteration)
        probs = eval(conv_model_robust, targeted_adv_images.to(device))
        preds = np.argmax(probs, axis=1)
        targeted_acc, _ , _ = eval_perf_multi(y_test.detach().cpu().numpy(), preds) 
        print(f"Robust model accuracy on targeted PGD images with target class {desired_class}: {targeted_acc}")

        targeted_examples = targeted_adv_images.detach().cpu().numpy()
        adv_list = list()
        for i in range(4):
            adv_list.append(TargetedAdvExample(x_test[i], targeted_examples[i]))
        adv_dict.update({desired_class: adv_list})

    graph_targeted_examples(adv_dict, pathname=f"./stats/targeted_adv_eps_{epsilon}_steps_{no_of_steps}_it_{koef_iteration}_2.jpg")
    quit()

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

    # Populate dict of adversarial examples
    adv_dict = dict()
    for i in range(len(y_test)):
        if y_test[i] in list(adv_dict.keys()):
            continue
        adv_dict.update({y_test[i]: [AdvExample(int(preds[i]), int(adv_preds[i]), x_test[i].detach().cpu().numpy(), adv_images[i].detach().cpu().numpy())]})
        if len(list(adv_dict.keys())) == 3:
            break
    
    # Show adversarial examples
    graph_adv_examples(adv_dict)

    # Show barplot with accuracies of robust and nonrobust models on normal vs adversarial data
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