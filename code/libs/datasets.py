from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T

from datasets import load_dataset


class HFImageDataset(torch.utils.data.Dataset):
    """
    A thin interface to bridge HF dataset and pytorch dataset 
    """
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        image = sample["image"].convert("RGB")
        return self.transform(image), int(sample["label"])


def normalize_to_neg_one_to_one(tensor):
    """Map [0, 1] pixels to [-1, 1] with a picklable, module-level callable."""
    return (tensor * 2) - 1


def build_dataset(name, split, data_folder, use_vae=False):
    """
    Training pixels use [-1, 1], or [0, 1] with a VAE. Test pixels use [0, 1].
    """
    assert name in ["MNIST", "MiniImageNet"]
    assert split in ["train", "test"]
    pixel_transforms = [T.ToTensor()]
    if not use_vae and split == "train":
        # normalize the input for DDPM / FM during training
        pixel_transforms.append(normalize_to_neg_one_to_one)

    # MNIST for digits
    if name == "MNIST":
        transform = T.Compose(pixel_transforms)
        dataset = torchvision.datasets.MNIST(
            data_folder,
            train=(split=="train"),
            transform=transform,
            download=True
        )
        num_classes = 10
        image_shape = [1, 28, 28]

    # Mini ImageNet (100 categorys, 500 samples each category for training)
    elif name == "MiniImageNet":
        if split == "train":
            transform = T.Compose([
                T.Resize(144),
                T.RandomCrop(128),
                T.RandomHorizontalFlip(),
                *pixel_transforms,
            ])
        else:
            # disable normalization during testing 
            transform = T.Compose([
                T.Resize(144),
                T.CenterCrop(128),
                *pixel_transforms,
            ])
        # hugging face datasets
        hf_dataset = load_dataset(
            "timm/mini-imagenet",
            split=split,
            cache_dir=data_folder,
        )
        dataset = HFImageDataset(hf_dataset, transform)
        num_classes = 100
        image_shape = [3, 128, 128]

    # New dataset can be added here

    else:
        raise ValueError('Dataset {:s} not supported.'.format(name))

    # Stable source identity for reference-statistics caching. The FID helper
    # adds the current transform and sample count when it constructs the key.
    dataset.fid_metadata = {
        "name": name,
        "split": split,
        "source": str(Path(data_folder).expanduser().resolve()),
        "torchvision": torchvision.__version__,
    }
    if name == "MiniImageNet":
        dataset.fid_metadata["hf_fingerprint"] = hf_dataset._fingerprint
    else:
        raw_folder = getattr(dataset, "raw_folder", None)
        if raw_folder is not None:
            dataset.fid_metadata["raw_files"] = [
                (p.name, p.stat().st_size, p.stat().st_mtime_ns)
                for p in sorted(Path(raw_folder).glob("*")) if p.is_file()
            ]
    return dataset, num_classes, image_shape


def build_dataloader(dataset, batch_size, num_workers, shuffle=True):
    """
    Create a dataloder for the target dataset
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        persistent_workers=(num_workers > 0),
    )
    return loader
