import torch.nn as nn
import torch.nn.functional as F

class CPPN1(nn.Module):

    def __init__(self):

      super(CPPN1, self).__init__()

      self.fc1 = nn.Linear(2, 64)
      self.fc2 = nn.Linear(64, 64)
      self.fce1 = nn.Linear(64, 64)
      self.fc3 = nn.Linear(64, 3)     
    

    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fce1(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.sigmoid(x)

        return x