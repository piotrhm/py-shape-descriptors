from descriptors.datasets.common import _load_dataset_from_path

from cv2 import contourArea, findContours, CHAIN_APPROX_SIMPLE, RETR_EXTERNAL
import numpy as np
import matplotlib.pyplot as plt


path = '/MPEG7dataset/'
ext = "*.gif"
images = _load_dataset_from_path(path=path, kind='face')

for img in images.values():
    contour, _ = findContours(img, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)    
    lenghts = [element.shape[0] for element in contour]
    idx = np.argmax(lenghts)

    contour = np.array(contour[idx])
    area = contourArea(contour)

    print(area/(img.shape[0]*img.shape[1]))

    contour = contour.reshape(-1, 2)
    N = contour.shape[0]

    dft = np.fft.fft2(contour)
    print(dft)

    print(contour.shape)
    print(dft.shape)

    x, y = dft.T
    plt.scatter(x, y)
    plt.show()

    break
