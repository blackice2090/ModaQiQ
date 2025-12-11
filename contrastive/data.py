import copy
from pathlib import Path
from itertools import islice

import h5py
import keras
import numpy as np
from PIL import Image
from tqdm.auto import tqdm


class ContrastiveDataset(keras.utils.PyDataset):
    IMAGE_FORMATS = set(Image.registered_extensions())

    def __init__(
        self,
        batch_size: int = 32,
        image_size: tuple[int, int] = (256, 256),
        seed: int = None,
        workers: int = 1,
        use_multiprocessing: bool = False,
        max_queue_size: int = 10,
    ):
        super().__init__(workers, use_multiprocessing, max_queue_size)
        self.epochs = 0
        self.batch_size = int(batch_size)
        self.image_size = tuple(map(int, image_size))
        self.seed = int(np.random.randint(2**32 - 1) if seed is None else seed)
        assert self.batch_size > 3, "batch_size is too small"
        assert self.batch_size % 2 == 0, "batch_size must be even"
        self.files: dict[str, Path]
        self.folders: list[str]
        self.cache_path: str | Path = None

    def load(self, data: str | Path | dict[str, str | Path]):
        if isinstance(data, dict):
            files = {
                folder: sorted(map(Path, files))
                for folder, files in data.items()
                if len(files) >= self.batch_size // 2
            }
        else:
            files = {
                folder.name: sorted(files)
                for folder in Path(data).iterdir()
                if folder.is_dir()
                for files in [
                    [
                        file
                        for file in folder.iterdir()
                        if file.suffix.lower() in self.IMAGE_FORMATS
                    ]
                ]
                if len(files) >= self.batch_size // 2
            }
        assert len(files) > 1, "batch_size is large or no enough images"
        self.files = files
        self.folders = sorted(self.files)
        return self

    def __getitem__(self, index):
        rng = self.rng(index)
        anchor = rng.randint(len(self.folders))
        paths = rng.choice(
            self.files[self.folders[anchor]],
            self.batch_size // 2,
            replace=False,
        ).tolist()
        labels = [anchor] * len(paths)

        while len(paths) < self.batch_size:
            chosen = rng.randint(len(self.folders))
            if chosen == anchor:
                continue
            files = self.files[self.folders[chosen]]
            paths.append(rng.choice(files))
            labels.append(chosen)

        images = self.load_images(paths)
        labels = np.asarray(labels, np.uint32)
        return (images, labels)

    def load_images(self, paths):
        if self.cache_path is not None:
            out = None
            with h5py.File(self.cache_path) as file:
                for i, path in enumerate(paths):
                    vector = file[str(path)][...]
                    if out is None:
                        out = np.empty_like(vector, shape=(len(paths), *vector.shape))
                    out[i] = vector
            return out
        images = np.empty((len(paths), *reversed(self.image_size), 3), np.float32)
        for i, path in enumerate(paths):
            image = Image.open(path)
            image = image_resize_with_pad(image, self.image_size)
            images[i] = image
        return images

    def cache(self, model, path="./features.h5"):
        files = (f for files in self.files.values() for f in files)
        with h5py.File(path, "a") as cache:
            with tqdm(total=len(self)) as progress:
                files = filter(lambda f: str(f) not in cache, files)
                while batch := list(islice(files, self.batch_size)):
                    images = self.load_images(batch)
                    features = model(images).numpy()
                    for element, vector in zip(batch, features):
                        cache[str(element)] = vector
                    progress.update()
                progress.update(progress.total - progress.n)
        self.cache_path = Path(path)

    @property
    def num_images(self):
        return sum(map(len, self.files.values()))

    def __len__(self):
        return self.num_images // self.batch_size

    def on_epoch_end(self):
        self.epochs += 1

    def split(self, *ratios):
        assert all(0 < r < 1 for r in ratios) and sum(ratios) < 1
        folders = self.folders.copy()
        self.rng(epochs=-1).shuffle(folders)
        splits = []
        for ratio in (*ratios, 1):
            count = int(len(self.folders) * ratio)
            selected, folders = folders[:count], folders[count:]
            other = copy.copy(self)
            other.load({k: v for k, v in self.files.items() if k in selected})
            splits.append(other)
        return tuple(splits)

    def rng(self, index=0, epochs=None):
        epochs = epochs or self.epochs
        seed = self.seed + epochs * self.num_images + index
        return np.random.RandomState(seed % (2**32 - 1))

    def iter_hard(self, model, same=True, threshold=0):
        plt_ds = copy.copy(self)
        plt_ds.cache_path = None
        plt_ds.load_images = lambda x: x
        for paths, labels in plt_ds:
            features = self.load_images(paths)
            similarity = model(features).numpy()
            labels = labels[None, :] == labels[:, None]
            labels = labels.astype("float32") + 1
            labels = np.tril(labels, -1) - 1
            if same:
                mask = (labels == 1) & (similarity > threshold)
            else:
                mask = (labels == 0) & (similarity < threshold)
            if mask.any():
                pairs = np.stack(np.where(mask), axis=-1)
                images = type(self).load_images(plt_ds, paths)
                yield images, similarity, pairs


def image_resize_with_pad(
    img: Image.Image,
    target_size: tuple[int, int],
) -> Image.Image:
    """Resize the image while preserving the aspect ratio with padding"""
    out = Image.new(img.mode, target_size)
    scale = min(out.size[0] / img.size[0], out.size[1] / img.size[1])
    size = (int(scale * img.size[0]), int(scale * img.size[1]))
    pad = ((out.size[0] - size[0]) // 2, (out.size[1] - size[1]) // 2)
    out.paste(img.resize(size), pad)
    return out
