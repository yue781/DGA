import torch
import numpy as np


def get_random_problems(batch_size, problem_size):
    depot_xy = torch.rand(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)

    if 10 <= problem_size < 20:
        demand_scaler = 20
    elif 20 <= problem_size < 40:
        demand_scaler = 30
    elif 40 <= problem_size < 70:
        demand_scaler = 40
    elif 70 <= problem_size < 120:
        demand_scaler = 50
    elif 120 <= problem_size < 150:
        demand_scaler = 60
    elif 150 <= problem_size < 180:
        demand_scaler = 70
    elif 180 <= problem_size < 250:
        demand_scaler = 80
    elif 250 <= problem_size < 400:
        demand_scaler = 90
    elif 400 <= problem_size < 750:
        demand_scaler = 100
    elif 750 <= problem_size <= 1000:
        demand_scaler = 250
    else:
        raise NotImplementedError

    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
    # shape: (batch, problem)

    return depot_xy, node_xy, node_demand


def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data
