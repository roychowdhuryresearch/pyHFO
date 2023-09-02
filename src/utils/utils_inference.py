import torch
from tqdm import tqdm
import numpy as np
from src.param.param_classifier import ParamModel, ParamPreprocessing
def load_ckpt(func, model_path):
    model_load = torch.load(model_path, map_location=torch.device('cpu'))
    model = model_load["model"]
    return ParamPreprocessing.from_dict(model_load["preprocessing"]), model

def load_model(model, state_dict, device):
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device).float()
    return model

def inference(model, x, device ,batch_size = 256):
    model.eval()
    x = torch.from_numpy(x).float()
    with torch.no_grad():
        res = batch_iterate(x, model, device ,batch_size)
    return np.squeeze(res > 0.5)

def normalize_img(a):
    batch_num = a.shape[0]
    h = a.shape[1]
    w = a.shape[2]
    a_reshape = a.reshape(batch_num, -1)
    a_min = torch.min(a_reshape, -1)[0].unsqueeze(1)
    a_max = torch.max(a_reshape, -1)[0].unsqueeze(1)
    c = 255.0 * (a_reshape - a_min)/(a_max - a_min)
    c = c.reshape(batch_num,h, w)
    return c

def batch_iterate(x, model, device ,batch_size = 256):
    res = []
    num = x.shape[0]
    num_batch = num // batch_size
    if num % batch_size:
        num_batch += 1
    for i in tqdm(range(num_batch)):
        start = i * batch_size
        end = min((i + 1) * batch_size, num)
        d = x[start:end].to(device)
        d[:, 0, :, :] = normalize_img(d[:, 0, :, :])
        d[:, 1, :, :] = normalize_img(d[:, 1, :, :])
        res.append(model(d).detach())
    res = torch.cat(res, 0)
    if device != "cpu":
        res = res.cpu()
    return res.numpy()

