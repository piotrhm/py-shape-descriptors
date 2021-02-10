from descriptors.datasets.common import _load_dataset_from_path
from descriptors.ellipticity.ellipticity import ellipticity

path = "" #path to dataset
ext = "*.gif"
images = _load_dataset_from_path(path=path, kind='face')

for img in images.values():
    print(ellipticity(img, "euclidean_ellipticity"))