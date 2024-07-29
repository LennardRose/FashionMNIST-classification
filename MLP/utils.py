from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch

plt.rcParams.update({'font.size': 14})

def get_FashionMNIST_dataset(train):
    """
    load FashionMNIST Data
    :param train: Set True if the Data ist for Training, False if for Test
    :return: The FashionMNIST Dataset as a Tensor
    """
    return FashionMNIST(
        root="data",
        train=train,
        download=True,
        transform=ToTensor()
    )


def get_data_loader(train, batch_size):
    """
    Return a Torch DataLoarder with the FashionMNIST dataset
    :param train:  Set True if the Data ist for Training, False if for Test
    :param batch_size: The Batch size for the loader to yield data
    :return: the dataloader
    """
    data = get_FashionMNIST_dataset(train=train) # if for training (train=true)
    return DataLoader(data,
                      batch_size,
                      shuffle=train)  # if for training (train=true), data needs to be shuffled)


def get_device(cuda):
    """
    get the device to train on, only if available!
    :param cuda: True if GPU, False if CPU
    :return:
    """
    if cuda and torch.cuda.is_available():
        # Clear cache if non-empty
        torch.cuda.empty_cache()
        # See which GPU has been allotted
        print(f"Using cuda device: {torch.cuda.get_device_name(torch.cuda.current_device())} for training")
        return "cuda"
    else:
        print("Using cpu for training")
        return "cpu"


def plot_training(epochs, learning_rate, hidden_dims, batch_size,
         training_losses, validation_losses, training_accuracies, validation_accuracies):
    """
    Plot 4 Subplots consisting of the Training and Validation losses and accuracies
    :param epochs:  The learning rate used during the training
    :param learning_rate: The learning rate used during the training
    :param hidden_dims: the hidden dimensions used for training, in case of cnn the output channels
    :param batch_size:  The learning rate used during the training
    :param training_losses: The training losses to display as a list
    :param validation_losses:  The Validation losses to display as a list
    :param training_accuracies:  The training accuracies to display as a list
    :param validation_accuracies:  The Validation accuracies to display as a list
    """

    fig, axs = plt.subplots(2, 2, figsize=(20, 10), constrained_layout=True)
    fig.suptitle(f'Epochs: {epochs} LR: {learning_rate} Hidden Layers: {hidden_dims} Batch Size: {batch_size}')

    # losses
    axs[0, 0].plot(range(epochs), training_losses)
    axs[0, 0].set_title("Training Losses")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].set_xlabel("Epochs")
    axs[1, 0].plot(range(epochs), validation_losses)
    axs[1, 0].set_title("Validation Losses")
    axs[1, 0].set_ylabel("Loss")
    axs[1, 0].set_xlabel("Epochs")

    # accuracies
    axs[0, 1].plot(range(epochs), training_accuracies)
    axs[0, 1].set_title("Training Accuracies")
    axs[0, 1].set_ylabel("Accuracy")
    axs[0, 1].set_xlabel("Epochs")
    axs[1, 1].plot(range(epochs), validation_accuracies)
    axs[1, 1].set_title("Validation Accuracies")
    axs[1, 1].set_ylabel("Accuracy")
    axs[1, 1].set_xlabel("Epochs")

    plt.show()


def plot_predictions(test_values, test_labels, predictions):
    """
    Plots the 10 test samples images as well as the predicted and the true label
    """
    fig, ax = plt.subplots(2, 5, figsize=(30, 15))

    i = 0
    for row in ax:
        for col in row:
            img = test_values[i].squeeze()
            col.set_title(f"True Label: {test_labels[i]}\nPredicted: {predictions[i]}")
            col.axis('off')
            col.imshow(img, cmap='gray')
            i += 1

    plt.show()
