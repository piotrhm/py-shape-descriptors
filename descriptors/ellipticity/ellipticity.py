from math import pi
from descriptors.utils.moments import moments
from descriptors.utils.shape_measurement import calc_centeroid, get_contour
from cv2 import contourArea, RETR_TREE, CHAIN_APPROX_SIMPLE, RETR_EXTERNAL

import numpy as np


def ellipticity(image, method='moment_invariants'):
    """
    There are 4 methods to define ellipticity:
    Moment invariants (in two similar versions): E_I
    Elliptic variance: E_V
    Euclidean ellipticity: E_E
    DFT: E_F

    :param image: np.ndarray, binary mask
    :return: float \in [0, 1]
    """
    if method == 'moment_invariants':
        m = moments(image)
        # I_1 affine moment invariant
        i_1 = (m['mu20'] * m['mu02'] - m['mu11'] ** 2) / m['mu00'] ** 4

        # I_1 for unit circle
        ic = 1 / (16 * pi ** 2)
        if i_1 <= ic:
            return i_1 / ic
        else:
            return ic / i_1
    elif method == 'elliptic_variance':
        N, contour = get_contour(image, RETR_TREE, CHAIN_APPROX_SIMPLE)
        centroid = np.array(calc_centeroid(contour))

        covariance_matrix = np.cov(contour.T)
        covariance_matrix_inv = np.linalg.inv(covariance_matrix)

        vec = contour - centroid
        vec_t = (contour - centroid).T

        eq = np.dot(vec, covariance_matrix_inv).dot(vec_t)
        sqrt = np.sqrt(np.abs(eq.diagonal()))
        mu = sqrt.sum() / N

        e_var = np.power(sqrt - mu, 2).sum() / (N * mu ** 2)
        norm_e_var = 1 / (1 + e_var)

        return norm_e_var
    elif method == 'euclidean_ellipticity':
        N, contour = get_contour(image, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
        area = contourArea(contour)

        min_med_error = np.inf
        best_parmas = np.array(5)
        count_valid = 0

        indices = [np.random.choice(contour.shape[0], 5, replace=False) for _ in range(100000)]
        for ind in indices:
            points = contour[ind]
            A = np.zeros((5, 5))
            for i, point in enumerate(points):
                x, y = point[0], point[1]
                A[i] = np.array([x ** 2, x * y, y ** 2, x, y])

            b = np.ones(5) * -1
            params = np.linalg.solve(A, b.T)

            # circle case
            eq = params[1] ** 2 - 4 * params[0] * params[2]
            if eq >= 0:
                continue

            count_valid += 1

            error = np.zeros(N)
            for i, point in enumerate(contour):
                x, y = point[0], point[1]
                error[i] = np.power(np.sum(np.array([x ** 2, x * y, y ** 2, x, y]) * params) + 1, 2)

            error.sort()
            med_error = np.median(error)
            if med_error < min_med_error:
                min_med_error = med_error
                best_parmas = params

            if count_valid > 50:
                break

        E = np.zeros(N)
        for i, point in enumerate(contour):
            x, y = point[0], point[1]
            E[i] = np.power(np.sum(np.array([x ** 2, x * y, y ** 2, x, y]) * best_parmas) + 1, 2)

        norm = E.sum() * np.sqrt(np.sqrt(area)) / (N)
        norm = 1 / (1 + norm)

        return norm
    elif method == 'dft':
        raise NotImplementedError("To be implemented")
    else:
        print("Wrong method.")
