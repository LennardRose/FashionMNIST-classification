import torch.nn as nn
import torch.optim as optim
from utils import *
from sklearn.metrics import accuracy_score
from training import train_model


def init_mlp(device, input_size, hidden_dims, output_size, batchnorm):
    """
    Initialize the MLP model with various amount of hidden dimensions
    :param device: the device to train on (gpu/cpu)
    :param input_size: dimensions of the input
    :param hidden_dims: list of hiddem dimensions e.g [3, 3, 3] for three layers with 3 neurons each
    :param output_size: the size of the output (dimension of output layer)
    :param batchnorm: Set True to set Batch-normalization layer between every Linear-Relu layer
    :return: The initialized MLP
    """
    # first layer flattens the input
    layers = [nn.Flatten()]

    # input layer
    previous_dimensions = input_size

    # hidden layer
    for current_dim in hidden_dims:
        layers.append(nn.Linear(previous_dimensions, current_dim, bias=True))
        previous_dimensions = current_dim
        layers.append(nn.ReLU())
        # append batchnorm after every layer
        if batchnorm:
            layers.append(nn.BatchNorm1d(current_dim))
            layers.append(nn.ReLU())

    # output layer
    layers.append(nn.Linear(previous_dimensions, output_size, bias=True))

    # bring everything together
    mlp = nn.Sequential(*layers)
    mlp.to(device)

    return mlp


def mlp_train(hidden_dims, epochs, batch_size, learning_rate, cuda, plot, verbose, batchnorm = False):
    """
    Trains a MLP model Steps: initialize model - train model - plot - return
    :param hidden_dims: List of integers to
    specify every hidden dimensions size
    :param learning_rate: Learning Rate for the Optimizer
    :param cuda: Use GPU (True) or CPU (False)
    :param epochs: Number of epochs to train the model
    :param batch_size: batch size for the training
    :param plot: set True to plot Losses/ accuracies of the training and validation
    :param verbose: set True to get Losses/ Accuracies output of every epoch
    :param batchnorm: Set True to set Batch-normalization layer between every Linear-Relu layer
    :return: the trained model, a tuple of the training and
    validation loss and a tuple of the training and validation accuracy
    """

    input_size = 28 * 28  # image dimension
    output_size = 10  # number of classes
    device = get_device(cuda)

    mlp = init_mlp(device=device,
                   input_size=input_size,
                   hidden_dims=hidden_dims,
                   output_size=output_size,
                   batchnorm=batchnorm)

    train_loss, train_acc, val_loss, val_acc = train_model(model=mlp,
                                                           criterion=nn.CrossEntropyLoss(),
                                                           optimizer=optim.SGD(params=mlp.parameters(),
                                                                               lr=learning_rate),
                                                           device=device,
                                                           batch_size=batch_size,
                                                           epochs=epochs,
                                                           verbose=verbose)

    if plot:
        plot_training(epochs=epochs,
                      learning_rate=learning_rate,
                      hidden_dims=hidden_dims,
                      batch_size=batch_size,
                      training_losses=train_loss,
                      training_accuracies=train_acc,
                      validation_losses=val_loss,
                      validation_accuracies=val_acc)

    return mlp, (train_loss, val_loss), (train_acc, val_acc)


def mlp_apply(mlp, test_indexes):
    """
    :param mlp: The model to predict the test samples
    :param test_indexes: list of Indexes to sample from the test set
    """
    # prepare data
    test_data = get_FashionMNIST_dataset(train=False)
    test_values = test_data.data[test_indexes]
    test_labels = test_data.targets[test_indexes]

    # predict and evaluate
    predictions = torch.argmax(mlp(test_values.float()), 1)
    accuracy = accuracy_score(test_labels, predictions) * 100

    # Print Results and display 10 data examples and labels
    print(f'Accuracy {accuracy}% on Test Samples: {test_indexes} (count: {len(test_indexes)})')

    plot_predictions(test_values=test_values,
                     test_labels=test_labels,
                     predictions=predictions)
