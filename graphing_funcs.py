import matplotlib.pyplot as plt
import numpy as np

def show_loss(loss):
    """
    Function for showcasing the loss over epochs
    Params:
        loss: list of losses over epochs
    """
    # Loss over the epochs graph
    fig = plt.figure(figsize=(16, 10))
    plt.plot(range(len(loss)), np.array(loss), label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss functions")
    plt.title("Loss function over the epochs")
    plt.legend()
    plt.show()

def show_train_accuracies(accs, epochs, name, path):
    """
    Function for showcasing the train accuracy of a model over epochs
    Params:
        acc: list of accuracies over epochs
        epochs: total number of epochs
        name: desired name of saved figure
        path: desired path of saved figure
    """

    # Train accuracy over the epochs graph 
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
    """
    Function for showcasing a weight matrix as a set of 28x28 pictures
    Params:
        weights: weight matrix 
    """
    fig = plt.figure(figsize=(16, 8))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow((weights[:, i].detach().cpu().numpy()).reshape(28, 28))
    plt.show()

def graph_stats(fc_times, fc_accs, conv_times, conv_accs):
    """
    Function for showcasing the accuracy/train time graph for fully connected models and convolutional models
    Params:
        fc_times: list of measured train times for fully connected models
        fc_accs: list of calculated test data accuracies for fully connected models
        conv_times: list of measured train times for convolutional models
        conv_accs: list of calculated test data accuracies for convolutional models
    """
    # Accuracy/train time graph for different models
    fig = plt.figure(figsize=(16,5))
    plt.plot(fc_accs, fc_times, 'r', label="FC model")
    plt.plot(conv_accs, conv_times, 'b', label="Conv model")
    plt.xlabel("Accuracies")
    plt.ylabel("Train times")
    plt.title("Accuracy/train time graph for FC and Conv models")
    plt.legend()
    plt.show()

def graph_details(fc_times, fc_accs, conv_times, conv_accs):
    """
    Function for showcasing the comparison of train times and accuracies for different models
    Params:
        fc_times: list of measured train times for fully connected models
        fc_accs: list of calculated test data accuracies for fully connected models
        conv_times: list of measured train times for convolutional models
        conv_accs: list of calculated test data accuracies for convolutional models
    """
    layer_num = [1, 2, 3]
    fig = plt.figure(figsize=(16,10))

    # Train time comparison for different models
    plt.subplot(2, 1, 1)
    plt.plot(layer_num, fc_times, 'r', label="FC model")
    plt.plot(layer_num, conv_times, 'b', label="Conv model")
    plt.xlabel("Number of layers")
    plt.ylabel("Train time")
    plt.title("Number of layers / Train time graph for FC and Conv models")
    plt.legend(loc="center right")

    # Accuracy comparison for different models
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
    """
    Function for showcasing the adversarial examples generated using different epsilon coefficients
    Params:
        adv_dict: dict[eps: list<advExample>]
    """
    fig = plt.figure(figsize=(20,10))
    length = len(adv_dict.keys())
    keys = list(adv_dict.keys())

    subfigs = fig.subfigures(nrows=length, ncols=1)
    if not isinstance(subfigs, np.ndarray):
        subfigs = [subfigs]

    # Show adversarial examples for each coefficient in seperate row
    for row, subfig in enumerate(subfigs):
        key = keys[row]
        adv_list = adv_dict[key]
        adv_cnt = len(adv_list)
        subfig.suptitle(f'Koeficijent: {key}')

        axs = subfig.subplots(nrows=1, ncols=adv_cnt * 2)
        i = 0

        # Show adversarial examples for given coefficient
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
    # plt.savefig('./stats/adversarial_examples_pgd.jpg')
    plt.show()

def graph_attack_accuracies(adv_accs):
    """
    Function for accuracies of a given model on adversarial examples
    Params:
        adv_accs: list of accuracies of a given model on adversarial examples
    """
    koefs = np.array(list(adv_accs.keys()))
    accs = np.array(list(adv_accs.values()))

    # Coefficient/Accuracy graph for given model on adversarial examples
    fig = plt.figure(figsize=(16,5))
    plt.plot(koefs, accs, 'b')
    plt.xlabel("Coefficients")
    plt.ylabel("Accuracies")
    plt.title("Coefficient/Accuracy graph for convolutional model")
    # plt.savefig('./stats/graph_attack_accuracies_pgd.jpg')
    plt.show()

def graph_adv_examples(adv_dict):
    """
    Function for showcasing the adversarial examples of a given model
    Params:
        adv_dict: dict[correctlabel: list<advExample>]
    """
    fig = plt.figure(figsize=(20,10))
    length = len(adv_dict.keys())
    keys = list(adv_dict.keys())

    subfigs = fig.subfigures(nrows=length, ncols=1)
    if not isinstance(subfigs, np.ndarray):
        subfigs = [subfigs]

    # Show adversarial examples for each correct label
    for row, subfig in enumerate(subfigs):
        key = keys[row]
        adv_list = adv_dict[key]
        adv_cnt = len(adv_list)
        subfig.suptitle(f'Ispravna oznaka: {key}', fontweight='bold')

        axs = subfig.subplots(nrows=1, ncols=adv_cnt * 2)
        i = 0

        for adv in adv_list:
            ax = axs[i]
            ax.plot()
            ax.imshow((adv.initial_img).reshape(28, 28))
            ax.set_title(f"Predikcija za originalnu sliku: {adv.inital_pred}")
            ax.axis('off')
            i += 1

            ax = axs[i]
            ax.plot()
            ax.imshow((adv.attacked_img).reshape(28, 28))
            ax.set_title(f"Predikcija za izmijenjenu sliku: {adv.attacked_pred}")
            ax.axis('off')
            i += 1

    plt.subplots_adjust(top=0.75)
    # plt.savefig('./stats/robust_adv_examples.jpg')
    # plt.savefig('./stats/nonrobust_adv_examples.jpg')
    plt.show()