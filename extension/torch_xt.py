import os 
import torch 
import torch.nn as nn

def set_device(x, device):
    use_cuda = False
    multi_gpu = False
    if len(device) == 1 and device[0] > 0:
        use_cuda = True 
    elif len(device) > 1:
        use_cuda = True 
        multi_gpu = True 

    # When input is tensor 
    if isinstance(x, torch.Tensor): 
        if use_cuda:
            x = x.cuda(device[0] - 1)
        else:
            x = x.cpu()
     # When input is model
    elif isinstance(x, nn.Module): 
        if use_cuda:
            if multi_gpu:
                devices = [i - 1 for i in device]
                torch.cuda.set_device(devices[0])
                x = nn.DataParallel(x, device_ids=devices).cuda()
            else: 
                torch.cuda.set_device(device[0] - 1)
                x.cuda(device[0] - 1)
        else: 
            x.cpu()
    # When input is tuple 
    elif type(x) is tuple or type(x) is list:
        x = list(x)
        for i in range(len(x)):
            x[i] = set_device(x[i], device)
        x = tuple(x) 

    return x 

def save_checkpoint(checkpoint_path, model, optimizer, learning_rate, iteration, verbose=False):
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, checkpoint_path)

    if verbose:
        print("Saving model and optimizer state at iteration {} to {}".format(
            iteration, checkpoint_path))

def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))

    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']

    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))

    return model, optimizer, learning_rate, iteration

class LossLog(object):
    def __init__(self):
        self.iteration = 0
        self.sum = 0.0
        self.num = 0

    def reset(self):
        self.sum = 0.0
        self.num = 0

    def add(self, x, batch_size):
        self.sum += batch_size*x
        self.num += batch_size
        self.iteration += 1

    def avg(self):
        return float(self.sum/self.num)