from utils import *
from sklearn.metrics import accuracy_score
from tqdm.notebook import trange


def train_model(model, criterion, optimizer, device, batch_size, epochs, verbose):
    """
    Executes the training for a given model, initializes the data, Trains the model and validates the training,
    records and outputs the progress
    :param model: The ANN Model to train
    :param criterion: The Loss/errorfunction
    :param optimizer: The method to update the weights
    :param device: The device to execute the training/validation on (cpu/gpu)
    :param batch_size:Batch size for the dataloader to yield
    :param epochs: number of epochs to train the model
    :param verbose: Set True to output every  epochs loss/ accuracy
    :return: Training and Validation losses and accuracies
    """
    training_dataloader = get_data_loader(train=True,
                                          batch_size=batch_size)

    test_dataloader = get_data_loader(train=False,
                                      batch_size=batch_size)

    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []

    for epoch in trange(epochs, desc="Epochs"):
        # training
        epoch_loss = 0
        prediction = []
        y_true = []

        model.train()
        for batch, labels in training_dataloader:
            torch.cuda.empty_cache()
            batch = batch.to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()

            output = model(batch)
            loss = criterion(output, labels)
            loss.backward()

            optimizer.step()

            predictions = output.argmax(dim=1)
            prediction += predictions.tolist()
            y_true += labels.tolist()
            epoch_loss += loss.item()

        # Loss and accuracy of current epoch
        avg_loss = epoch_loss / len(training_dataloader)
        accuracy = accuracy_score(y_true, prediction) * 100
        training_losses.append(avg_loss)
        training_accuracies.append(accuracy)

        # validation
        epoch_loss = 0
        prediction = []
        y_true = []

        model.eval()
        with torch.no_grad():
            for batch, labels in test_dataloader:
                torch.cuda.empty_cache()
                batch = batch.to(device)
                labels = labels.long().to(device)

                output = model(batch)
                loss = criterion(output, labels)

                predictions = output.argmax(dim=1)
                prediction += predictions.tolist()
                y_true += labels.tolist()
                epoch_loss += loss.item()

        # Loss and accuracy of current epoch
        val_loss = epoch_loss / len(test_dataloader)
        val_accuracy = accuracy_score(y_true, prediction) * 100
        validation_losses.append(val_loss)
        validation_accuracies.append(val_accuracy)

        #output
        if verbose:
            print(f"Epoch {epoch + 1} / {epochs}")
            print(f"Training Loss: {round(avg_loss, 3)} - Accuracy: {round(accuracy, 2)}%")
            print(f"Validation Loss: {round(val_loss, 3)} - Accuracy: {round(val_accuracy, 2)}%")

    return training_losses, training_accuracies, validation_losses, validation_accuracies



