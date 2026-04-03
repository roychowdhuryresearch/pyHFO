import torch
from tqdm import tqdm
import numpy as np
from src.param.param_classifier import ParamModel, ParamPreprocessing

def load_ckpt(func, model_path):
    model_load = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model = model_load["model"]
    return ParamPreprocessing.from_dict(model_load["preprocessing"]), model

def load_preprocessing_param(param_dict):
    return ParamPreprocessing.from_dict(param_dict)

def load_model(model, state_dict, device):
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device).float()
    return model

def inference(model, x, device ,batch_size = 256, threshold = 0.5):
    model.eval()
    with torch.inference_mode():
        res = batch_iterate(x, model, device, batch_size)
    return np.squeeze(res > threshold)

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
    num = x.shape[0]
    if num == 0:
        return np.array([], dtype=np.float32)
    num_batch = num // batch_size
    if num % batch_size:
        num_batch += 1

    first_batch = torch.from_numpy(x[:min(batch_size, num)]).float()
    first_result = model(_prepare_batch(first_batch, device)).detach().cpu().numpy()
    output_shape = first_result.shape
    outputs = np.empty((num,) + tuple(output_shape[1:]), dtype=np.float32)

    start = 0
    end = min(batch_size, num)
    outputs[start:end] = first_result

    for i in tqdm(range(num_batch)):
        if i == 0:
            continue
        start = i * batch_size
        end = min((i + 1) * batch_size, num)
        batch = torch.from_numpy(x[start:end]).float()
        outputs[start:end] = model(_prepare_batch(batch, device)).detach().cpu().numpy()
    return outputs


def _prepare_batch(batch, device):
    d = batch.to(device, non_blocking=(device != "cpu"))
    d[:, 0, :, :] = normalize_img(d[:, 0, :, :])
    d[:, 1, :, :] = normalize_img(d[:, 1, :, :])
    if d.shape[1] == 3:
        d[:, 2, :, :] = normalize_img(d[:, 2, :, :])
    return d
