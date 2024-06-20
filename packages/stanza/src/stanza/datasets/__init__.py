from typing import Generic, TypeVar, Mapping, Tuple, Callable

from stanza.dataclasses import dataclass
from stanza.data import Data
from stanza.data.transform import Transform
from stanza.data.normalizer import Normalizer
from stanza.util.registry import Registry, from_module

import jax
import logging
logger = logging.getLogger("stanza.datasets")

T = TypeVar('T')

@dataclass
class Dataset(Generic[T]):
    splits: Mapping[str, Data[T]]
    normalizers: Mapping[str, Callable[[], Normalizer[T]]]
    transforms: Mapping[str, Callable[[], Transform[T]]]

DatasetRegistry = Registry

@dataclass
class ImageDataset(Dataset[jax.Array]):
    pass

@dataclass
class ImageClassDataset(Dataset[Tuple[jax.Array, jax.Array]]):
    classes: list[str]

    def as_image_dataset(self) -> ImageDataset:
        def map_normalizer(normalizer_builder):
            def mapped(*args, **kwargs):
                return normalizer_builder(*args, **kwargs).map(lambda x: x[0])
            return mapped

        return ImageDataset(
            splits={k: v.map(lambda x: x[0]) for k, v in self.splits.items()},
            normalizers={
                k: map_normalizer(v)
                for k, v in self.normalizers.items()
            },
            transforms={}
        )

@dataclass
class EnvDataset(Dataset[T], Generic[T]):
    def create_env(self):
        raise NotImplementedError()

env_datasets : DatasetRegistry[EnvDataset] = DatasetRegistry()
env_datasets.extend("pusht", from_module(".pusht", "datasets"))

image_class_datasets : DatasetRegistry[ImageClassDataset] = DatasetRegistry[Dataset]()
"""Datasets containing (image, label) pairs,
where label is one-hot encoded."""
image_class_datasets.extend("mnist", from_module(".mnist", "datasets"))
image_class_datasets.extend("cifar", from_module(".cifar", "datasets"))

__all__ = [
    "Dataset",
    "DatasetRegistry",
    "image_label_datasets"
]
