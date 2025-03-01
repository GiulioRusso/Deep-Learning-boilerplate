import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: adjust the model class based on your task

class MyNetwork(nn.Module):
    """
    My Neural Network Class
    """

    def __init__(self,
                 num_classes: int = 10):
        """
        __init__ method: run one when instantiating the object
        """

        super(MyNetwork, self).__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)


    def forward(self,
                input: torch.Tensor) -> torch.Tensor:
        """
        forward method: directly call a method in the class when an instance name is called

        :param input: input data
        :return: output: network output
        """

        # applying first convolutional layer followed by ReLU and pooling
        x = self.pool(F.relu(self.conv1(input)))

        # applying second convolutional layer followed by ReLU and pooling
        x = self.pool(F.relu(self.conv2(x)))

        # flattening the output for the fully connected layer
        x = torch.flatten(x, 1)  # Flatten all dimensions except the batch

        # first fully connected layer with ReLU
        x = F.relu(self.fc1(x))

        # output layer with no activation
        output = self.fc2(x)

        return output  # B x num_classes
