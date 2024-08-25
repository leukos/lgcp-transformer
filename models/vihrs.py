import torch
import torch.nn as nn

class VihrsNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initializes Vihrs network for esitmating

        :param nec: number of input layers, typically 3 for color images.
        :param nc: number of ouput layers, typicall 3 or 1 depending on task.
        :param nef: base multiplier for the convolutional filter banks.
        """
        super(VihrsNN, self).__init__()

        # Use convs on the L function
        self.l_transform = nn.Sequential(
            nn.Conv1d(input_dim, 64, 7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5),

            nn.Conv1d(64, 64, 7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5),

            nn.Conv1d(64, 64, kernel_size=5),
            nn.ReLU(),
            nn.Flatten()
        )

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5),
            nn.Flatten()
        )

        # Estimator
        self.estimator = nn.Sequential(
            nn.Linear(961, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )


    def forward(self, x):
        """
        The forward function simply applies
        """
        l, n = x
        out = self.l_transform(l)
        # out2 = self.conv2d(map)
        inp = torch.cat([out,  n], axis=1)
        out = self.estimator(inp)
        return out
