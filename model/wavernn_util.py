import torch
import torch.nn.functional as F

def cumproduct(x):
    product = 1
    y = []
    for el in x:
        product *= el
        y.append(product)

    return y

def mol_distribution(y, log_scale_min=None):
    """
    Sample from discretized mixture of logistic distributions
    Args:
        y (Tensor): B x C x T
        log_scale_min (float): Log scale minimum value
    Returns:
        Tensor: sample in range of [-1, 1].
    """
    if log_scale_min is None:
        log_scale_min = torch.log(torch.tensor(1e-14)).item() #float(np.log(1e-14))
    assert y.size(1) % 3 == 0
    nr_mix = y.size(1) // 3

    # B x T x C
    y = y.transpose(1, 2)
    logit_probs = y[:, :, :nr_mix]

    # sample mixture indicator from softmax
    temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-5, 1.0 - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=-1)

    # (B, T) -> (B, T, nr_mix)
    one_hot = F.one_hot(argmax, nr_mix).float()
    # select logistic parameters
    means = torch.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, dim=-1)
    log_scales = torch.clamp(torch.sum(
        y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, dim=-1), min=log_scale_min)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = means.data.new(means.size()).uniform_(1e-5, 1.0 - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))

    x = torch.clamp(torch.clamp(x, min=-1.), max=1.)

    return x

def label2float(x, bits):
    return 2*x/(2**bits - 1.0) - 1.0

def decode_mu_law(y, mu, from_labels=True):
    if from_labels: y = label2float(y, torch.log2(mu))
    mu = mu - 1
    x = torch.sign(y)/mu*((1 + mu)**torch.abs(y) - 1)
    return x

def xfade_and_unfold(y, target, overlap):
    num_folds = y.size(0)
    length = y.size(1)

    target = length - 2*overlap
    total_len = num_folds*(target + overlap) + overlap

    silence_len = overlap // 2
    fade_len = overlap - silence_len
    silence = y.new_zeros(silence_len, dtype=torch.float64)
    linear = y.new_ones(silence_len, dtype=torch.float64)

    t = torch.linspace(-1, 1, steps=fade_len, dtype=torch.float64, device=y.device)
    fade_in = torch.sqrt(0.5 * (1 + t))
    fade_out = torch.sqrt(0.5 * (1 - t))

    fade_in = torch.cat((silence, fade_in))
    fade_out = torch.cat((linear, fade_out))

    y[:, :overlap] *= fade_in
    y[:, -overlap:] *= fade_out

    unfolded = y.new_zeros(total_len, dtype=torch.float64)
    
    for i in range(num_folds):
        start = i*(target + overlap)
        end = start + target + 2*overlap
        unfolded[start:end] += y[i]

    return unfolded 