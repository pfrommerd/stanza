from stanza.datasets import DatasetRegistry, ImageDataset
from stanza.data import IOData

from . import util as du

class CelebAData(IOData):
    def __init__(self, path):
        self._path = path
        super().__init__()
    
    def _fetch(self, idx):
        return du.read_image(self._path / f"{idx+1:06d}.jpg")

    def __len__(self):
        return 202599

def _load_celeb_a(quiet=False, **kwargs):
    data_path = du.cache_path("celeb_a") / "img_align_celeba.zip"
    du.download(data_path,
        gdrive_id="1Yo6KZFeQeuplQ_fvqvqAei0WouFbjKjT",
        md5="00d2c5bc6d35e252742224ab0c1e8fcb",
        job_name="CelebA"
    )
    extract_path = du.cache_path("celeb_a") / "images"
    if not extract_path.exists():
        du.extract_to(data_path, extract_path,
            job_name="CelebA",
            quiet=quiet,
            strip_folder="img_align_celeba"
        )
    
    train_data = CelebAData(extract_path)

    return ImageDataset(splits={"train": train_data})


registry = DatasetRegistry()
registry.register("celeb_a", _load_celeb_a)