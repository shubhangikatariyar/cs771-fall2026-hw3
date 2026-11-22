# Modified from:
# https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py

"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from .inception import InceptionV3

IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        with Image.open(path) as image:
            img = image.convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_activations(
    files, model,
    target_size=None, batch_size=50, dims=2048, device="cpu", num_workers=1
):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size; the final partial batch is included.
    -- target_size : Target image size. FID is known to be sensitive to blur.
                     A solution is to always resize real / generated images into
                     a fixed target size.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if len(files) < 2:
        raise ValueError("FID statistics require at least two images")

    if batch_size > len(files):
        print(
            (
                "Warning: batch size is bigger than the data size. "
                "Setting batch size to data size"
            )
        )
        batch_size = len(files)

    if target_size is None:
        transforms=TF.ToTensor()
    else:
        transforms=TF.Compose([
            TF.Resize(target_size),
            TF.ToTensor()
        ])
    dataset = ImagePathDataset(files, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx : start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(
    files, model,
    target_size=None, batch_size=50, dims=2048, device="cpu", num_workers=1
):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- target_size : Target image size. FID is known to be sensitive to blur.
                     A solution is to always resize real / generated images into
                     a fixed target size.
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(
        files, model, target_size, batch_size, dims, device, num_workers
    )
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(
    path, model, target_size, batch_size, dims, device, num_workers=1
):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as f:
            m, s = _validate_statistics(f["mu"], f["sigma"], dims)
    else:
        files = _image_files(path)

        m, s = calculate_activation_statistics(
            files, model, target_size, batch_size, dims, device, num_workers
        )

    return m, s

def compute_statistics_from_loader(loader, model, device="cpu"):
    """Expect image tensors in [0, 1], with shape [B, C, H, W]."""
    model.eval()
    features = []

    with torch.inference_mode():
        for batch in tqdm(loader):
            if isinstance(batch, dict):
                images = batch["image"]
            elif isinstance(batch, (tuple, list)):
                images = batch[0]  # Ignore class labels for standard FID.
            else:
                images = batch

            images = images.to(device=device, dtype=torch.float32)
            if images.ndim != 4:
                raise ValueError("FID expects images with shape [B, C, H, W]")
            if not torch.isfinite(images).all() or (images < 0).any() or (images > 1).any():
                raise ValueError("FID expects finite image pixels in [0, 1]")

            # gray -> RGB if needed
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
            elif images.shape[1] != 3:
                raise ValueError("FID expects one or three image channels")

            pred = model(images)[0]
            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            features.append(pred)

    if not features:
        raise ValueError("Cannot compute FID statistics for an empty dataset")

    activations = np.concatenate(features, axis=0).astype(np.float64)
    if len(activations) < 2:
        raise ValueError("FID covariance requires at least two images")

    # compute stats
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma

def calculate_fid_given_paths(
    paths, target_size, batch_size, device, dims, num_workers=1
):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError("Invalid path: %s" % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = compute_statistics_of_path(
        paths[0], model, target_size, batch_size, dims, device, num_workers
    )
    m2, s2 = compute_statistics_of_path(
        paths[1], model, target_size, batch_size, dims, device, num_workers
    )
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value

def calculate_fid_between_path_loader(
    loader, path, target_size, batch_size, device, dims, num_workers=1
):
    """Compare generated images on disk with a reference DataLoader."""
    if not os.path.exists(path):
        raise RuntimeError("Invalid path: %s" % path)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = compute_statistics_of_path(
        path, model, target_size, batch_size, dims, device, num_workers
    )
    m2, s2 = compute_statistics_from_loader(
        loader, model, device
        # other parameter are set within the loader itself
    )
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value

def save_fid_stats(
    data_pointer, fid_file, target_size, batch_size, device, dims, num_workers=1
):
    """Save statistics from a directory, .npz file, or DataLoader."""
    
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    print(f"Saving statistics to {fid_file}")

    if isinstance(data_pointer, torch.utils.data.DataLoader):
        m, s = compute_statistics_from_loader(
            data_pointer, model, device
        )
    else:
        if not os.path.exists(data_pointer):
            raise RuntimeError("Invalid path: %s" % data_pointer)
        m, s = compute_statistics_of_path(
            data_pointer, model, target_size, batch_size, dims, device, num_workers
        )
    # save to npz file
    _validate_statistics(m, s, dims)
    metadata = None
    if (not isinstance(data_pointer, torch.utils.data.DataLoader)
            or hasattr(data_pointer.dataset, "fid_metadata")):
        metadata = _cache_metadata(data_pointer, target_size, dims)
    _write_statistics(fid_file, m, s, metadata=metadata)
    return


def _image_files(path):
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"FID image directory does not exist: {path}")
    return sorted(p for p in path.rglob("*")
                  if p.is_file() and p.suffix.lower().lstrip(".") in IMAGE_EXTENSIONS)


def _validate_statistics(mu, sigma, dims):
    if mu.shape != (dims,) or sigma.shape != (dims, dims):
        raise ValueError(f"FID statistics must have feature dimension {dims}")
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all():
        raise ValueError("FID statistics must be finite")
    return mu, sigma


def _write_statistics(path, mu, sigma, metadata=None):
    """Publish a complete .npz atomically, including for concurrent evaluations."""
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("FID statistics path must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"mu": mu, "sigma": sigma}
    if metadata is not None:
        arrays["metadata"] = metadata
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
            temporary = Path(f.name)
            np.savez_compressed(f, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _cache_metadata(data_pointer, target_size, dims):
    """Identify the reference distribution, independently of generated samples."""
    if isinstance(data_pointer, torch.utils.data.DataLoader):
        dataset = data_pointer.dataset
        reference = getattr(dataset, "fid_metadata", None)
        if reference is None:
            raise ValueError("Cached DataLoaders require dataset.fid_metadata identifying the source and split")
        sampler = data_pointer.sampler
        if (data_pointer.drop_last
                or not isinstance(sampler, (torch.utils.data.SequentialSampler, torch.utils.data.RandomSampler))
                or (isinstance(sampler, torch.utils.data.RandomSampler) and sampler.replacement)
                or len(sampler) != len(dataset)):
            raise ValueError("Cached FID requires a full-dataset loader without replacement or drop_last")
        reference = dict(reference, count=len(dataset),
                         transform=repr(getattr(dataset, "transform", None)))
    else:
        path = Path(data_pointer).resolve()
        files = [path] if path.suffix == ".npz" else _image_files(path)
        manifest = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
        reference = {"source": str(path), "manifest": manifest}
    return json.dumps({
        "version": 1,
        "extractor": "pytorch-fid-inception-v3",
        "dims": dims,
        "target_size": target_size,
        "reference": reference,
    }, sort_keys=True)


def compute_fid_score(
    data_pointer, output_path,
    stats_path=None, target_size=None, num_workers=None, dims=2048, batch_size=50,
    stats_dir=None,
):
    """Compare a reference loader/path with generated images.

    stats_path selects an explicit cache file; stats_dir selects automatic,
    metadata-keyed filenames. With neither, reference statistics are not cached.
    Custom loader datasets must supply a JSON-serializable fid_metadata dict
    identifying their source, split, and revision to enable caching.
    """

    # FID score is computed using CPU to ensure numerical stability
    device = torch.device("cpu")

    if num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0

    metadata = None
    cache = None
    reference_stats = None
    if stats_path is not None or stats_dir is not None:
        metadata = _cache_metadata(data_pointer, target_size, dims)
        digest = hashlib.sha256(metadata.encode()).hexdigest()[:16]
        cache = Path(stats_path) if stats_path is not None else Path(stats_dir) / f"fid-{digest}.npz"
        if cache.suffix != ".npz":
            raise ValueError("FID statistics path must end in .npz")
        if cache.is_file():
            try:
                with np.load(cache, allow_pickle=False) as f:
                    if "metadata" in f and f["metadata"].item() == metadata:
                        reference_stats = _validate_statistics(f["mu"], f["sigma"], dims)
            except (ValueError, KeyError, OSError, EOFError, zipfile.BadZipFile):
                pass  # Recompute incompatible or incomplete caches.
        if reference_stats is not None:
            print(f"Using cached FID statistics: {cache}")

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    if reference_stats is None:
        if isinstance(data_pointer, torch.utils.data.DataLoader):
            reference_stats = compute_statistics_from_loader(data_pointer, model, device)
        else:
            reference_stats = compute_statistics_of_path(
                data_pointer, model, target_size, batch_size, dims, device, num_workers
            )
        _validate_statistics(*reference_stats, dims)
        if cache is not None:
            _write_statistics(cache, *reference_stats, metadata=metadata)
            print(f"Saved FID statistics: {cache}")

    generated_stats = compute_statistics_of_path(
        output_path, model, target_size, batch_size, dims, device, num_workers
    )
    return calculate_frechet_distance(*reference_stats, *generated_stats)


