import torch

def np2th(weights, conv=False):
    """
    [height, width, input_channel, output_channel] -> [output_channel, input_channel, height, width]
    """
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    
    return torch.from_numpy(weights)