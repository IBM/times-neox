# modified from source code from https://github.com/vgurev/pytorch-transformer-ts/blob/main/lag-gpt/aug.py 
# and https://github.com/vafl/gluon-ts/blob/ts_embeddings/src/gluonts/nursery/ts_embeddings/pt_augmentation.py



import numpy as np
import torch, random


@torch.no_grad()
def window_warp(x, window_ratio, scales):
    """https://halshs.archives-ouvertes.fr/halshs-01357973/document"""

    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio * x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)

    window_starts = np.random.randint(
        low=1, high=x.shape[1] - warp_size - 1, size=(x.shape[0])
    ).astype(int)
    window_ends = (window_starts + warp_size).astype(int)

    ret = torch.zeros_like(x)
    for i, pat in enumerate(x):
        start_seg = pat[: window_starts[i]].cpu().numpy()
        window_seg = np.interp(
            np.linspace(
                0,
                warp_size - 1,
                num=int(warp_size * warp_scales[i]),
            ),
            window_steps,
            pat[window_starts[i] : window_ends[i]].cpu().numpy(),
        )
        end_seg = pat[window_ends[i]:].cpu().numpy()
        warped = np.concatenate((start_seg, window_seg, end_seg))
        warp = np.interp(
            np.arange(x.shape[1]),
            np.linspace(0, x.shape[1] - 1.0, num=warped.size),
            warped,
        )
        ret[i] = torch.from_numpy(warp).float().to(x.device)
    return ret



@torch.no_grad()
def window_slice(x, reduce_ratio):
    """https://halshs.archives-ouvertes.fr/halshs-01357973/document"""

    target_len = np.ceil(reduce_ratio * x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(
        low=0, high=x.shape[1] - target_len, size=(x.shape[0])
    ).astype(int)
    ends = (target_len + starts).astype(int)

    ret = torch.zeros_like(x)
    for i, pat in enumerate(x):
        
        warp = np.interp(
            np.linspace(0, target_len, num=x.shape[1]),
            np.arange(target_len),
            pat[starts[i] : ends[i]].cpu().numpy(),
        ).T
        ret[i] = torch.from_numpy(warp).float().to(x.device)

    return ret

@torch.no_grad()
def time_warp(x, sigma, knot):
    from scipy.interpolate import CubicSpline

    orig_steps = np.arange(x.shape[1])
    random_warps = np.random.normal(
        loc=1.0,
        scale=sigma,
        size=(x.shape[0], knot + 2),
    )
    warp_steps = np.linspace(0, x.shape[1] - 1.0, num=knot + 2)

    ret = torch.zeros_like(x)
    for i, pat in enumerate(x):
        
        time_warp = CubicSpline(
            warp_steps,
            warp_steps * random_warps[i],
        )(orig_steps)
        scale = (x.shape[1] - 1) / time_warp[-1]
        wrap = np.interp(
            orig_steps,
            np.clip(scale * time_warp, 0, x.shape[1] - 1),
            pat.cpu().numpy(),
        ).T
        ret[i] = torch.from_numpy(wrap).float().to(x.device)

    return ret


@torch.no_grad()
def magnitude_warp(x, sigma, knot):
    from scipy.interpolate import CubicSpline

    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(
        loc=1.0,
        scale=sigma,
        size=(x.shape[0], knot + 2),
    ) 
    warp_steps = np.linspace(0, x.shape[1] - 1.0, num=knot + 2)
    ret = torch.zeros_like(x)
    for i, pat in enumerate(x):
        warper = CubicSpline(warp_steps, random_warps[i])(orig_steps)
        mean = torch.mean(pat, dim = -1, keepdim = True)
        ret[i] = (pat - mean) * torch.from_numpy(warper).float().to(x.device) + mean

    return ret


@torch.no_grad()
def permutation(x, max_segments, seg_mode = None):

    orig_steps = np.arange(x.shape[1])
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = torch.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            
            if seg_mode == "random":
                split_points = np.random.choice(
                    x.shape[1] - 2, num_segs[i] - 1, replace=False
                )
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            elif seg_mode is None:
                splits = np.array_split(orig_steps, num_segs[i])
            else:
                raise ValueError(f"seg_mod {seg_mode} is not supported by permutation augmentation.")
            
            random.shuffle(splits)
            warp = np.concatenate(splits).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret


@torch.no_grad()
def jitter(x, sigma):
    '''
    Gaussain noise scaled by std of the series
    '''
    std = torch.std(x, dim = -1)[..., np.newaxis]
    return x + std * torch.normal(
        mean=0.0, std=sigma, 
        size=x.shape, device=x.device)


@torch.no_grad()
def rotation(x):
    flip_index = torch.multinomial(
            torch.tensor([0.5, 0.5], dtype=x.dtype, device=x.device),
            num_samples=x.shape[0],replacement=True,
    )
    ones = torch.ones((x.shape[0]), device=x.device)
    flip = torch.where(flip_index == 0, -ones, ones)
    return flip[..., np.newaxis] * x


@torch.no_grad()
def freq_mask(xy, rate=0.1, dim=1):
    
    xy_copy = xy
    xy_f = torch.fft.rfft(xy, dim=dim)
    m = torch.empty(xy_f.shape, dtype = xy.dtype).uniform_() < rate

    freal = xy_f.real.masked_fill(m, 0)
    fimag = xy_f.imag.masked_fill(m, 0)
    xy_f = torch.complex(freal, fimag)
    xy = torch.fft.irfft(xy_f, dim=dim)

    if xy_copy.shape[dim] != xy.shape[dim]:
        xy = torch.cat([xy_copy[:, 0:1, ...], xy], dim)

    return xy


@torch.no_grad()
def freq_mix(xy, rate=0.1, dim=1):
    
    xy_copy = xy
    xy_f = torch.fft.rfft(xy, dim=dim)

    m = torch.empty(xy_f.shape, dtype = xy.dtype).uniform_() < rate
    amp = abs(xy_f)
    _, index = amp.sort(dim=dim, descending=True)
    dominant_mask = index > 2
    m = torch.bitwise_and(m, dominant_mask)
    freal = xy_f.real.masked_fill(m, 0)
    fimag = xy_f.imag.masked_fill(m, 0)

    b_idx = np.arange(xy.shape[0])
    np.random.shuffle(b_idx)
    xy2 = xy_copy[b_idx]
    xy2_f = torch.fft.rfft(xy2, dim=dim)

    m = torch.bitwise_not(m)
    freal2 = xy2_f.real.masked_fill(m, 0)
    fimag2 = xy2_f.imag.masked_fill(m, 0)

    freal += freal2
    fimag += fimag2

    xy_f = torch.complex(freal, fimag)
    xy = torch.fft.irfft(xy_f, dim=dim)

    if xy_copy.shape[dim] != xy.shape[dim]:
        xy = torch.cat([xy_copy[:, 0:1, ...], xy], 1)

    return xy



augmentation_map = {
    "freq_mask": freq_mask,
    "freq_mix": freq_mix,
    "jitter": jitter,
    "rotation": rotation,
    "permutation": permutation,
    "magnitude_warp": magnitude_warp,
    "time_warp": time_warp,
    "window_slice": window_slice,
    "window_warp": window_warp,
}


def augmentaiton_factory(opt):

    augmentations = []
    values = []
    index = 0
    for key, value in opt.items():
        aug = augmentation_map[key]
        weight = value["weight"]
        options = value.get("options", {})

        augmentations.append((aug, options))
        values.append(weight)

    values = np.array(values)
    weights = values / np.sum(values)
    return augmentations, weights


class AugmentationIterator:
    '''
    Transformation of iterator that apply with probability "prob" augmentations.
    Below an example of opt parameter.
    {
        "prob": 1.0,
        "transforms": {
            "freq_mask": {
                "weight": 0.5,
                "options": {
                    "rate": 0.01
                }
            },
            "freq_mix": {
                "weight": 0.5,
                "options": {
                    "rate": 0.01
                }
            }
        }
    }
    "options" are parameters to agmentation functions
    '''
    def __init__(self, opt, iterator):
        self.iterator = iterator
        self.prob = opt["prob"]
        transforms = opt["transforms"]
        self.augmentations, self.weights = augmentaiton_factory(transforms)


    def __next__(self):

        batch = next(self.iterator)
        
        if random.random() < self.prob:
            x, y = batch["past_target"], batch["future_target"]
            prev = batch["past_target"]
            x_len, y_len = x.shape[-1], y.shape[-1]
            xy = torch.cat([x, y], dim = -1)

            aug_index = np.random.choice(np.arange(len(self.weights)), size = 1, p=self.weights)
            augmentation, pars = self.augmentations[aug_index[0]]
            transformed_batch = augmentation(xy, **pars)
            batch["past_target"], batch["future_target"] = torch.split(transformed_batch, [x_len, y_len], dim = -1)
            
        return batch

