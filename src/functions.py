import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from collections import Counter
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

def drop_column(df, atr_name: str) -> None:
    df.drop(atr_name, axis=1, inplace=True)


def plot_class_distribution(y_labels, title:str) -> None:
    # Count the occurrences of each class
    class_counts = y_labels.value_counts()  
  
    # Plot the distribution  
    plt.figure(figsize=(10,6))  
    plt.bar(class_counts.index, class_counts.values, alpha=0.5)
      
    plt.xticks(class_counts.index)  # Set x-ticks to be class labels  
    plt.xlabel('Classes')  
    plt.ylabel('Amount of samples')  
    plt.title(title) 
    plt.tight_layout() 
    plt.show()


def plot_class_distribution_tf(dataset, title: str) -> None:
    # Extract the labels from the dataset
    y_labels = [label.item() for _, label in dataset]

    # Count the occurrences of each class
    class_counts = dict(Counter(y_labels))

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values(), alpha=0.5)

    plt.xticks(list(class_counts.keys()))  # Set x-ticks to be class labels
    plt.xlabel('Classes')
    plt.ylabel('Amount of samples')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def parse_data_from_input(filename):
  """
  Parses the images and labels from a CSV file

  Args:
    filename (string): path to the CSV file

  Returns:
    images, labels: tuple of numpy arrays containing the images and labels
  """
  with open(filename) as file:
    csv_reader = csv.reader(file, delimiter=',')
    next(csv_reader)
    images = []
    labels = []

    for row in csv_reader:
        images.append([int(num) for num in row[1:]])
        labels.append(int(row[0]))

    labels = np.array(labels, dtype=np.float64)
    images = np.array(images, dtype=np.float64).reshape((-1, 28, 28))

    return images, labels

def adjust_class_labels(label):
  """Subtracts one from the label when greater than 9."""
  if label >= 10:
      label -= 1

  return label


def train(
        model,
        loss_func,
        optimizer,
        training_dataloader,
        validation_dataloader,
        epochs=10,
        enable_logging=False,
        device="cpu",
):
    """Trains the neural net."""

    # Preallocating the arrays for losses and accuracies
    loss_history_train = [0] * epochs
    accuracy_history_train = [0] * epochs
    loss_history_valid = [0] * epochs
    accuracy_history_valid = [0] * epochs

    # Launching the algorithm
    for epoch in range(epochs):

        # Enabling training mode
        model.train()

        # Considering each batch for the current epoch
        for x_batch, y_batch in training_dataloader:
            # Moving data to GPU
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Generating predictions for the batch
            model_predictions = model(x_batch)

            # Computing the loss
            loss = loss_func(model_predictions, y_batch.long())

            # Computing gradients
            loss.backward()

            # Updating parameters using gradients
            optimizer.step()

            # Resetting the gradients to zero
            optimizer.zero_grad()

            # Adding the batch-level loss and accuracy to history
            loss_history_train[epoch] += loss.item() * y_batch.size(0)
            is_correct = (
                    torch.argmax(model_predictions, dim=1) == y_batch
            ).float()
            accuracy_history_train[epoch] += is_correct.sum().cpu()

            # Computing epoch-level loss and accuracy
        loss_history_train[epoch] /= len(training_dataloader.dataset)
        accuracy_history_train[epoch] /= len(training_dataloader.dataset)

        # Enabling evaluation mode
        model.eval()

        # Testing the CNN on the validation set
        with torch.no_grad():

            # Considering each batch for the current epoch
            for x_batch, y_batch in validation_dataloader:
                # Moving data to GPU
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                # Generating predictions for the batch
                model_predictions = model(x_batch)

                # Computing the loss
                loss = loss_func(model_predictions, y_batch.long())

                # Adding the batch-level loss and accuracy to history
                loss_history_valid[epoch] += loss.item() * y_batch.size(0)
                is_correct = (
                        torch.argmax(model_predictions, dim=1) == y_batch
                ).float()
                accuracy_history_valid[epoch] += is_correct.sum().cpu()

        # Computing epoch-level loss and accuracy
        loss_history_valid[epoch] /= len(validation_dataloader.dataset)
        accuracy_history_valid[epoch] /= len(validation_dataloader.dataset)

        # Logging the training process
        if enable_logging:
            print(
                "Epoch {}/{}\n"
                "train_loss = {:.4f}, train_accuracy = {:.4f} | "
                "valid_loss = {:.4f}, valid_accuracy = {:.4f}".format(
                    epoch + 1,
                    epochs,
                    loss_history_train[epoch],
                    accuracy_history_train[epoch],
                    loss_history_valid[epoch],
                    accuracy_history_valid[epoch],
                )
            )

    return (
        model,
        loss_history_train,
        accuracy_history_train,
        loss_history_valid,
        accuracy_history_valid,
    )


def plot_train_val_loss(loss_history_train, loss_history_valid):
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(loss_history_train, lw=3)
    ax.plot(loss_history_valid, lw=3)
    ax.set_xlabel('Epoch', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.legend(['Training loss', 'Validation loss'], fontsize=15)
    plt.tight_layout()
    plt.show()

def plot_train_val_acc(accuracy_history_train, accuracy_history_valid):
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(accuracy_history_train, lw=3)
    ax.plot(accuracy_history_valid, lw=3)
    ax.set_xlabel('Epoch', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.legend(['Training accuracy', 'Validation accuracy'], fontsize=15)

    plt.tight_layout()
    plt.show()


def evaluate_test_by_batch(model, testing_dataloader, device="cpu"):
    """Computes predictions and accuracy for the Dataloader."""
    # Initializing the counter for accuracy
    accuracy_test = 0
    # Initializing a list for storing predictions
    test_predictions = []
    # Setting the model to the evaluation model
    model.eval()
    # Computing accuracy and predictions
    with torch.no_grad():
        for x_batch, y_batch in testing_dataloader:
            # Moving data to GPU
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            # Computing predictions for the batch
            test_batch_predictions = torch.argmax(model(x_batch), dim=1)
            # Adding the batch predictions to the prediction list
            test_predictions.append(test_batch_predictions)
            # Computing the test accuracy
            is_correct = (test_batch_predictions == y_batch).float()
            accuracy_test += is_correct.sum().cpu()

    # Transforming a list of tensors into one tensor
    test_predictions_tensor = torch.cat(test_predictions).cpu()
    # Finishing computing test accuracy
    accuracy_test /= len(testing_dataloader.dataset)

    return accuracy_test, test_predictions_tensor

def save_whole_model(model, filename):
    torch.save(model, filename)

def load_whole_model(filename):
    model = torch.load(filename)
    model.eval() # Set the model to evaluation mode
    return model

# Function to load an image
def load_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert image to grayscale
    return image


# Function to preprocess the image
def preprocess_image(image):
    # Resize image and convert it to tensor
    preprocess = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    image = preprocess(image)
    image = image.reshape(1, 1, 28, 28)  # Reshape image to match model's input shape
    return image


def load_and_evaluate_model(test_data_path, model_path):
    # Load the model
    model = load_whole_model(model_path)

    # Load the test data
    test_df = pd.read_csv(test_data_path)

    # Preprocess the data
    y_test = test_df['label']
    drop_column(test_df, 'label')
    features_test = test_df.values
    features_test = features_test.reshape(-1, 1, 28, 28)
    features_test_scaled = features_test / 255
    y_test = torch.from_numpy(y_test.values).float()
    x_test = torch.from_numpy(features_test_scaled).float()

    # Create a DataLoader for the test data
    testing_dataset = TensorDataset(x_test, y_test)
    testing_dataloader = DataLoader(testing_dataset, batch_size=64, shuffle=False)

    # Evaluate the model
    accuracy_test, predictions_test = evaluate_test_by_batch(model, testing_dataloader)
    print(f"Test accuracy: {accuracy_test:.4f}")

    # Display some predictions
    nums = np.random.randint(low=0, high=x_test.shape[0], size=10)
    fig = plt.figure(figsize=(13, 5))
    for i, num in enumerate(nums):
        ax = fig.add_subplot(2, 5, i + 1)
        ax.set_xticks([]);
        ax.set_yticks([])
        ax.imshow(x_test[num].numpy().reshape(28, 28), cmap="gray_r")
        ax.set_title(f"True label = {y_test[num].int()} \n Predicted label = {predictions_test[num]}")
    plt.suptitle("Random 10 predictions for test set", fontsize=20)
    plt.tight_layout()
    plt.show()

def create_model_1():
    model = nn.Sequential()
    model.add_module("conv1", nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2))
    model.add_module("relu1", nn.ReLU())
    model.add_module("pool1", nn.MaxPool2d(kernel_size=2))
    model.add_module("conv2", nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2))
    model.add_module("relu2", nn.ReLU())
    model.add_module("pool2", nn.MaxPool2d(kernel_size=2))
    model.add_module("flatten", nn.Flatten())
    model.add_module('fc1', nn.Linear(3136, 1024))
    model.add_module('relu3', nn.ReLU())
    model.add_module('dropout', nn.Dropout(p=0.5))
    model.add_module('fc2', nn.Linear(1024, 24))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer

def create_model_2():
    model = nn.Sequential()
    model.add_module("conv1", nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2))
    model.add_module("relu1", nn.ReLU())
    model.add_module("pool1", nn.MaxPool2d(kernel_size=2))
    model.add_module("conv2", nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2))
    model.add_module("relu2", nn.ReLU())
    model.add_module("pool2", nn.MaxPool2d(kernel_size=2))
    model.add_module("flatten", nn.Flatten())
    model.add_module('fc1', nn.Linear(3136, 1024))
    model.add_module('relu3', nn.ReLU())
    model.add_module('dropout', nn.Dropout(p=0.4))  # Changed dropout
    model.add_module('fc2', nn.Linear(1024, 32))  # Changed number of output neurons
    optimizer = optim.Adam(model.parameters(), lr=0.002)  # Changed learning rate
    return model, optimizer

def create_model_3():
    model = nn.Sequential()
    model.add_module("conv1", nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2))
    model.add_module("relu1", nn.ReLU())
    model.add_module("pool1", nn.MaxPool2d(kernel_size=2))
    model.add_module("conv2", nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2))
    model.add_module("relu2", nn.ReLU())
    model.add_module("pool2", nn.MaxPool2d(kernel_size=2))
    model.add_module("flatten", nn.Flatten())
    model.add_module('fc1', nn.Linear(3136, 512))  # Changed number of neurons
    model.add_module('relu3', nn.ReLU())
    model.add_module('dropout', nn.Dropout(p=0.5))
    model.add_module('fc2', nn.Linear(512, 24))  # Changed number of input neurons
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer


def train_and_select_best_model(training_dataloader, validation_dataloader, DEVICE='cpu', RANDOM_SEED=0):
    models = [create_model_1(), create_model_2(), create_model_3()]
    best_model = None
    best_accuracy = 0
    i = 1
    best_num = 0
    for i, (model, optimizer) in enumerate(models):
        torch.manual_seed(RANDOM_SEED)
        model.to(DEVICE)
        model, loss_history_train, accuracy_history_train, loss_history_valid, accuracy_history_valid = train(
            model=model,
            loss_func=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            training_dataloader=training_dataloader,
            validation_dataloader=validation_dataloader,
            epochs=8,
            enable_logging=True,
        )

        # Calculate validation accuracy
        validation_accuracy = accuracy_history_valid[-1]
        print(f"Model {i + 1} validation accuracy: {validation_accuracy:.4f}")

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_model = model
            best_num = i

        # Plotting training/validation loss
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(loss_history_train, lw=3)
        ax.plot(loss_history_valid, lw=3)
        ax.set_xlabel('Epoch', size=15)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.legend(['Training loss', 'Validation loss'], fontsize=15)

        # Plotting training/validation accuracy
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(accuracy_history_train, lw=3)
        ax.plot(accuracy_history_valid, lw=3)
        ax.set_xlabel('Epoch', size=15)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.legend(['Training accuracy', 'Validation accuracy'], fontsize=15)

        fig.suptitle(f'Training and Validation Loss and Accuracy with model {i}', fontsize=20)
        i += 1
        plt.tight_layout()
        plt.show()

    print(f"Best model is Model {best_num} with validation accuracy: {best_accuracy:.4f}")

    return best_model


