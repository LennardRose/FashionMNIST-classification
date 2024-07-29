import torch.nn as nn
import torch.optim as optim
from Neural_Networks_Assignments.SS23_Assignment_3.utils import *
from sklearn.metrics import accuracy_score
from Neural_Networks_Assignments.SS23_Assignment_3.training import train_model


def init_cnn(device, input_channel, out_channels, output_size):
    """

    :param device:
    :param input_channel:
    :param out_channels:
    :param output_size:
    :return:
    """
    layers = []

    # input layer
    previous_channels = input_channel

    # hidden layer
    for out_channel in out_channels:
        layer = nn.Sequential(
            nn.Conv2d(previous_channels, out_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        layers.append(layer)
        previous_channels = out_channel

    # prepare for linear
    layers.append(nn.Flatten())
    scalefactor = int(28 / 2 ** len(out_channels)) ** 2

    # two final linear layers
    layers.append(nn.Linear(in_features=scalefactor * previous_channels, out_features=100))
    layers.append(nn.Linear(in_features=100, out_features=output_size))

    # bring everything together
    cnn = nn.Sequential(*layers)
    cnn.to(device)

    return cnn


def cnn_train(out_channels, epochs, batch_size, learning_rate, cuda, plot, verbose):
    """
    Trains a CNN model Steps: initialize model - train model - plot - return
    :param out_channels: List of integers to
    specify every  layers channels
    :param learning_rate: Learning Rate for the Optimizer
    :param cuda: Use GPU (True) or CPU (False)
    :param epochs: Number of epochs to train the model
    :param batch_size: batch size for the training
    :param plot: set True to plot Losses/ accuracies of the training and validation
    :param verbose: set True to get Losses/ Accuracies output of every epoch
    :return: the trained model, a tuple of the training and
    validation loss and a tuple of the training and validation accuracy
    """
    device = get_device(cuda)
    input_channel = 1
    output_size = 10  # number of classes

    cnn = init_cnn(device=device,
                   input_channel=input_channel,
                   output_size=output_size,
                   out_channels=out_channels)

    train_loss, train_acc, val_loss, val_acc = train_model(model=cnn,
                                                           criterion=nn.CrossEntropyLoss(),
                                                           optimizer=optim.SGD(params=cnn.parameters(),
                                                                               lr=learning_rate),
                                                           device=device,
                                                           batch_size=batch_size,
                                                           epochs=epochs,
                                                           verbose=verbose)

    if plot:
        plot_training(epochs=epochs,
                      learning_rate=learning_rate,
                      hidden_dims=out_channels,
                      batch_size=batch_size,
                      training_losses=train_loss,
                      training_accuracies=train_acc,
                      validation_losses=val_loss,
                      validation_accuracies=val_acc)

    return cnn, (train_loss, val_loss), (train_acc, val_acc)


def cnn_apply(cnn_model, test_indexes):
    """

    :param cnn_model: The model to predict the test samples
    :param test_indexes: list of Indexes to sample from the test set
    """
    # prepare data
    test_data = get_FashionMNIST_dataset(train=False)
    test_values = test_data.data[test_indexes]
    test_values = test_values[:, None, ...]  # explicitly add the singleton channel dimension to match models dimensions
    test_labels = test_data.targets[test_indexes]

    # predict and evaluate
    predictions = torch.argmax(cnn_model(test_values.float()), 1)
    accuracy = accuracy_score(test_labels, predictions) * 100

    # Print Results and display 10 data examples and labels
    print(f'Accuracy {accuracy}% on Test Samples: {test_indexes} (count: {len(test_indexes)})')

    plot_predictions(test_values=test_values,
                     test_labels=test_labels,
                     predictions=predictions)
