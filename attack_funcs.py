import torch
import numpy as np

from util import class_to_onehot
from train_util import get_loss, eval, eval_perf_multi, eval_after_epoch
from graphing_funcs import graph_attack, graph_attack_accuracies
from AdvExample import AdvExample

def attack_fgsm(model, images, labels, eps=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    adv_examples = images.clone().detach().to(device)
    adv_examples.requires_grad = True

    probs = model(adv_examples)
    loss = get_loss(probs, labels)

    model.zero_grad()
    loss.backward()

    data_grad = adv_examples.grad.data

    adv_examples = adv_examples.detach() + eps * data_grad.sign()
    adv_examples = torch.clamp(adv_examples, min=0, max=1).detach()

    return adv_examples

def attack_pgd(model, images, labels, eps=0.3, koef_it=0.05, steps=7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    adv_examples = images.clone().detach()

    for _ in range(steps):
        adv_examples = adv_examples.to(device)
        adv_examples.requires_grad = True

        probs = model(adv_examples)
        loss = get_loss(probs, labels)

        model.zero_grad()
        loss.backward()

        data_grad = adv_examples.grad.data

        adv_examples = adv_examples.detach() + koef_it * data_grad.sign()
        delta = torch.clamp(adv_examples - images, min=-eps, max=eps)
        adv_examples = torch.clamp(images + delta, min=0, max=1).detach()

    return adv_examples

def attack_model_pgd(model, x_test, y_test, eps_list=[0.1, 0.2, 0.3], koefs_it=[0.01, 0.03, 0.05]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    adv_dict = dict()
    adv_accs = dict()

    for eps, koef_it in zip(eps_list, koefs_it):
        torch.cuda.empty_cache()
        print(f"Attacking with coefficient: {eps}...")
        permutations = torch.randperm(len(x_test))
        x_test = x_test.detach()[permutations]
        y_test = y_test.detach()[permutations]

        y_test_oh = class_to_onehot(y_test.detach().cpu())
        y_test_oh = torch.tensor(y_test_oh).to(device)

        probs = model(x_test.to(device))
        y_pred = np.argmax(probs.detach().cpu().numpy(), axis=1)

        adv_examples = attack_pgd(model, x_test, y_test_oh, eps)

        att_probs = eval(model, adv_examples)
        attacked_pred = np.argmax(att_probs, axis=1)  

        acc, _ , _ = eval_perf_multi(y_test.detach().cpu().numpy(), attacked_pred)

        adv_list = list()
        for i in range(len(y_pred)):
            if y_pred[i] != attacked_pred[i]:
                adv_list.append(AdvExample(int(y_pred[i]), int(attacked_pred[i]), x_test[i].detach().cpu().numpy(), adv_examples[i].detach().cpu().numpy()))
                if len(adv_list) == 3: break

        adv_dict.update({eps: adv_list})
        adv_accs.update({eps: acc})

        print(f"Finished attacking with coefficient: {eps}...")
    
    model.train()

    graph_attack(adv_dict)
    graph_attack_accuracies(adv_accs)
        

def attack_model_fgsm(model, x_test, y_test, eps_list=[0.1, 0.2, 0.3]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    adv_dict = dict()
    adv_accs = dict()

    for eps in eps_list:
        torch.cuda.empty_cache()
        print(f"Attacking with coefficient: {eps}...")
        permutations = torch.randperm(len(x_test))
        x_test = x_test.detach()[permutations]
        y_test = y_test.detach()[permutations]

        y_test_oh = class_to_onehot(y_test.detach().cpu())
        y_test_oh = torch.tensor(y_test_oh).to(device)

        probs = model(x_test.to(device))
        y_pred = np.argmax(probs.detach().cpu().numpy(), axis=1)

        adv_examples = attack_fgsm(model, x_test, y_test_oh, eps)

        att_probs = eval(model, adv_examples)
        attacked_pred = np.argmax(att_probs, axis=1)  

        acc, _ , _ = eval_perf_multi(y_test.detach().cpu().numpy(), attacked_pred)

        adv_list = list()
        for i in range(len(y_pred)):
            if y_pred[i] != attacked_pred[i]:
                adv_list.append(AdvExample(int(y_pred[i]), int(attacked_pred[i]), x_test[i].detach().cpu().numpy(), adv_examples[i].detach().cpu().numpy()))
                if len(adv_list) == 3: break

        adv_dict.update({eps: adv_list})
        adv_accs.update({eps: acc})

        print(f"Finished attacking with coefficient: {eps}...")
    
    model.train()

    graph_attack(adv_dict)
    graph_attack_accuracies(adv_accs)

def train_robust(model, X, Y, param_niter=1000, param_delta=1e-2, param_lambda=1e-3, batch_size=1000, epoch_print=100, conv=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            adv_images = attack_pgd(model, x, y, eps=0.3)
            adv_probs = model(adv_images)
            adv_loss = get_loss(adv_probs, y) + (param_lambda * model.get_norm() if not conv else 0)
            temp_loss.append(adv_loss.detach().cpu().item())
            opt.zero_grad()
            adv_loss.backward()
            opt.step()

            #print("Batch = " + str(i))
            probs = model(x)
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