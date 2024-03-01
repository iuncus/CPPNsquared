import torch
import torch.nn as nn
import torch.nn.functional as F
from src.util import get_normalised_coordinate_grid
from src.util import normalize
from PIL import Image
import numpy as np
import random


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
    
def spatial_coords(array, layer):

    coords = []
    row1 = []
    row2 = []
    for i in range(array.shape[0]):
        row2.append([layer+1, i])
    for i in range(array.shape[1]):
        row1.append([layer, i])
            
    array = array.flatten("C").tolist()
    for i in row2:
        for j in row1:
            temp = []
            temp.extend(j)
            temp.extend(i)
            coords.append((temp))

    return coords, array


class CPPN1training():
  def run_loop():

    device = 'cuda'
    num_steps = 2000
    batch_size = 20000
    learn_rate = 0.01
    momentum = 0.9
    num_channels = 3
    image_shape = (512, 644)

    target_im_path = "ACNMW_ACNMW_DA000182-001.jpg"
    target_im = Image.open(target_im_path).convert("RGB")
    resized_im = target_im.resize(image_shape)

    all_coords = get_normalised_coordinate_grid(image_shape)
    all_coords = torch.tensor(all_coords, device = device, dtype=torch.float32)

    all_pix_vals = np.reshape(resized_im, [-1, num_channels]) / 255
    all_pix_vals = torch.tensor(all_pix_vals, device = device, dtype = torch.float32)

    cppn1 = CPPN1()
    cppn1.to(device)
    cppn1.requires_grad_

    optimizer = torch.optim.Adam(cppn1.parameters(), lr = learn_rate)
    criterion = nn.MSELoss(reduction = "mean")


    num_coords = all_coords.shape[0]
    coords_indexes = list(range(0, num_coords))
    losses = []
    running_loss = 0.0
    best_loss = 100000

    for i in range(num_steps):
      optimizer.zero_grad()
      cppn1.zero_grad()

      training_indexes = torch.tensor(np.array(random.sample(coords_indexes, batch_size)))
      training_coords = all_coords[training_indexes]
      pix_val_batch = all_pix_vals[training_indexes]
      approx_pix_val = cppn1(training_coords)
      loss = criterion(approx_pix_val, pix_val_batch)
      running_loss += loss.item()
      losses.append(loss.item())
      if running_loss < best_loss:
        best_loss = running_loss
      loss.backward()
      optimizer.step

    target_coords = []
    target_weights = []
    index = 0
    for name, param in cppn1.named_parameters():
      
      # print(name)
      if name.endswith(".weight"):
          
        # print(index)
        # print(param)
        temp_layer = param.cpu().detach().numpy() # need to learn more about gradients and why they are required
        # print(temp_layer)

        temp_coords, temp_weights = spatial_coords(temp_layer, index)
        temp_coords = torch.tensor(temp_coords, device=device, dtype=torch.float32)
        normal = temp_coords
        # print(temp_coords)
        for i in range(4):
          # print(i)
          normal[:,i] = normalize(temp_coords[:, i], i, cppn1)
          # normal = torch.nan_to_num(normal, nan = 0)
        # if index == 3:
          # print(temp_coords)
          # print(normal)
          # print(normal)
        target_coords.extend(normal.tolist())
        target_weights.extend(temp_weights)

        index += 1
          # print(fc1.shape[0])

    return(target_weights)