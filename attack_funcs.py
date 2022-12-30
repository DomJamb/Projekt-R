import torch
import numpy as np

from util import class_to_onehot
from train_util import get_loss, eval, eval_perf_multi
from graphing_funcs import graph_attack, graph_attack_accuracies
from AdvExample import AdvExample

def attack_fgsm(image, data_grad, koef=0.3):
    data_grad = torch.sign(data_grad)
    attacked_image = image + data_grad * koef
    return torch.clamp(attacked_image, min=0, max=1)

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

    y_test_oh = class_to_onehot(y_test.detach().cpu())
    y_test_oh = torch.tensor(y_test_oh).to(device)

    adv_dict = dict()
    adv_accs = dict()

    for eps, koef_it in zip(eps_list, koefs_it):
        print(f"Attacking with coefficient: {eps}...")
        permutations = torch.randperm(len(x_test))
        x_test = x_test.detach()[permutations]
        y_test_oh = y_test_oh.detach()[permutations]

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
        

def attack_model_test(model, x_test, y_test, koefs=[0.1, 0.2, 0.3], adv_cnt=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    adv_dict = dict()
    adv_accs = dict()

    for koef in koefs:
        print(f"Attacking with coefficient: {koef}...")
        permutations = torch.randperm(len(x_test))
        x_test = x_test.detach()[permutations]
        y_test = y_test.detach()[permutations]

        adv_list = list()
        acc = 0
        for input, correct_class in zip(x_test, y_test):
            input = input.to(device)
            input.requires_grad = True

            probs = model.forward(input)
            y_pred = np.argmax(probs.detach().cpu().numpy(), axis=1)
            
            if correct_class != torch.tensor(y_pred):
                continue
            
            correct_class_oh = np.zeros(10)
            correct_class_oh[int(correct_class)] = 1

            loss = get_loss(probs, torch.tensor(correct_class_oh).to(device))

            model.zero_grad()
            loss.backward()

            data_grad = input.grad.data
            attacked_image = attack_fgsm(input, data_grad, koef)

            tprobs = eval(model, attacked_image.to(device))
            attacked_pred = np.argmax(tprobs, axis=1)

            if torch.tensor(attacked_pred) == correct_class:
                #print("Still classifies correctly...")
                acc += 1
                continue

            if len(adv_list) < adv_cnt:
                adv_list.append(AdvExample(int(y_pred), int(attacked_pred), input.detach().cpu().numpy(), attacked_image.detach().cpu().numpy()))
    
        print(f"Finished attacking with coefficient: {koef}...")
        adv_dict.update({koef: adv_list})
        acc = acc / len(x_test)
        adv_accs.update({koef: acc})
        
    model.train()

    graph_attack(adv_dict)
    graph_attack_accuracies(adv_accs)