import matplotlib.pyplot as plt
import numpy as np

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
    #plt.savefig(path + name)
    plt.show()

def show_weights(weights):
    fig = plt.figure(figsize=(16, 8))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow((weights[:, i].detach().cpu().numpy()).reshape(28, 28))
    plt.show()

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

def graph_attack(adv_dict):
    fig = plt.figure(figsize=(20,10))
    length = len(adv_dict.keys())
    keys = list(adv_dict.keys())

    subfigs = fig.subfigures(nrows=length, ncols=1)
    if not isinstance(subfigs, np.ndarray):
        subfigs = [subfigs]

    for row, subfig in enumerate(subfigs):
        key = keys[row]
        adv_list = adv_dict[key]
        adv_cnt = len(adv_list)
        subfig.suptitle(f'Koeficijent: {key}')

        axs = subfig.subplots(nrows=1, ncols=adv_cnt * 2)
        i = 0

        for adv in adv_list:    
            ax = axs[i]
            ax.plot()
            ax.imshow((adv.initial_img).reshape(28, 28))
            ax.set_title(f"Originalna predikcija: {adv.inital_pred}")
            ax.axis('off')
            i += 1

            ax = axs[i]
            ax.plot()
            ax.imshow((adv.attacked_img).reshape(28, 28))
            ax.set_title(f"Izmijenjena predikcija: {adv.attacked_pred}")
            ax.axis('off')
            i += 1

    plt.subplots_adjust(top=0.75)
    #plt.savefig('./stats/adversarial_examples.jpg')
    plt.show()

def graph_attack_accuracies(adv_accs):
    koefs = np.array(list(adv_accs.keys()))
    accs = np.array(list(adv_accs.values()))

    fig = plt.figure(figsize=(16,5))
    plt.plot(koefs, accs, 'b')
    plt.xlabel("Coefficients")
    plt.ylabel("Accuracies")
    plt.title("Coefficient/Accuracy graph for convolutional model")
    #plt.savefig('./stats/graph_attack_accuracies.jpg')
    plt.show()