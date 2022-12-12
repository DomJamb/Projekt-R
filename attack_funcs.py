import torch
import numpy as np

from train_util import get_loss
from train_util import eval
from graphing_funcs import graph_attack, graph_attack_accuracies
from AdvExample import AdvExample

def attack_fgsm(image, data_grad, koef=0.3):
    data_grad = torch.sign(data_grad)
    attacked_image = image + data_grad * koef
    return torch.clamp(attacked_image, min=0, max=1)

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