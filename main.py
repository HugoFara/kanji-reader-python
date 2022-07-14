"""
A kanji recognition script.
"""

import argparse
import json
import numpy as np
import cv2
#import matplotlib.pyplot as plt

# Show debugging informations
_DEBUG = True

# True: Register new kanji instead of analysing one
_LEARNING = True

# Square root of the Pixels Per Kanji, a kanji will be fitted into an image of size PPKxPPK
PPK = 256


def parse_args():
    """Parse the differebt argument, prepare globals."""
    global _LEARNING
    parser = argparse.ArgumentParser(description='Code for Feature Detection.')
    parser.add_argument(
        '--o', help='Path to output result'
    )
    parser.add_argument(
        '-l', '--learning', help='Learning mode instead of recognition',
        action="store_true"
    )
    parser.add_argument(
        '--strokes', help='Strokes count for the kanji.', type=int
    )
    parser.add_argument(
        '--name', help='Only in learning mode. Helper to register kanji.'
    )
    parser.add_argument(
        'image', help='Path to input image.', default='ai.jpeg'
    )
    args = parser.parse_args()
    _LEARNING = bool(args.learning)
    return args


def get_image(filepath):
    """Parse the commend line arguments and get images from it."""
    img1 = cv2.imread(cv2.samples.findFile(filepath), cv2.IMREAD_GRAYSCALE)
    if img1 is None:
        raise FileNotFoundError('Could not open or find the images!')
    return img1


def format_image(image):
    """
    Format an image in standard input.

    Apply threshold, crop to the interesting part, and resize in 512x512.
    """
    cv2.imshow("Input", image)
    threshold = 255 / 100 * 60
    _, image = cv2.threshold(image, threshold, 255, cv2.THRESH_OTSU)
    cv2.imshow("Threshed", image)
    # Crop to an interesting bounding box
    # [[min_x, min_y], [max_x, max_y]]
    bbox = [[image.shape[1], image.shape[0]], [0, 0]]
    for i, line in enumerate(image):
        for j, val in enumerate(line):
            if not val:
                bbox = [
                    [min(bbox[0][0], j), min(bbox[0][1], i)],
                    [max(bbox[1][0], j), max(bbox[1][1], i)]
                ]
    center = (
        (bbox[0][0] + bbox[1][0]) // 2, (bbox[0][1] + bbox[1][1]) // 2
    )
    extend = max(bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1]) // 2
    image = image[
        slice(center[0] - extend, center[0] + extend),
        slice(center[1] - extend, center[1] + extend)
    ]
    cv2.imshow("Cropped", image)
    image = cv2.resize(image, (PPK, PPK), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Resized", image)
    return image


def morphological_simplify(image):
    """
    Try to process the image using morphological transfomations.

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.

    Returns
    -------
    im_bw : TYPE
        DESCRIPTION.

    """
    # Using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    # Erosion then dilatation
    im_bw = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    im_bw = cv2.dilate(im_bw, kernel, iterations=2)
    cv2.imshow("Lines", im_bw)
    cv2.waitKey(0)
    return im_bw


def hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x)
    hessian_mat = np.empty((x.ndim, x.ndim) + x.shape, dtype=np.float64)
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k)
        for l, grad_kl in enumerate(tmp_grad):
            hessian_mat[k, l, :, :] = grad_kl
    return hessian_mat


def gradient_simplify(image):
    """
    Use image gradient to compute main lines.

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.

    Returns
    -------
    im_bw : TYPE
        DESCRIPTION.

    """
    # Using gaussian blur
    kernel = [145, 145]
    im_bw = cv2.GaussianBlur(image, kernel, 0)
    cv2.imshow("Gaussian blur", im_bw)
    # Gradient over dx and dy (in two lists)
    # dim = np.gradient(im_bw)
    hess = hessian(im_bw)

    shaped = hess[0, 0] * hess[1, 1] - hess[0, 1] * hess[1, 0]
    for i, line in enumerate(shaped):
        for j, col in enumerate(line):
            shaped[i, j] = 0
            # Use only minima
            if col < 0:
                shaped[i, j] = 255
            # shaped[i, j] = min(255, max(0, shaped[i, j]))
    cv2.imshow("Hessianned", shaped)

    cv2.waitKey(0)
    return im_bw


def grow_spheres(spheres, image, threshold):
    """
    Grow spheres to find keypoints.

    Parameters
    ----------
    spheres : list[list[int, int, int]]
        Spheres to grow.
    image : TYPE
        Input formatted image.
    threshold : int
        Threshold value of black and white.

    Returns
    -------
    spheres : list[list[int, int, int]]
        Remaining spheres after growing.

    """
    n_spheres = 0
    growing = True
    maxxed = []
    while growing or n_spheres != len(spheres):
        n_spheres = len(spheres)
        print(n_spheres)
        # Grow spheres
        growing = False
        for s in spheres:
            if s[:2] in maxxed:
                continue
            dist = s[2] + 1
            grow = True
            for i, line in enumerate(image):
                if not grow or abs(i - s[1]) > dist:
                    continue
                for j, val in enumerate(line):
                    if not grow or abs(j - s[0]) > dist:
                        continue
                    if np.sqrt((i - s[1]) ** 2 + (j - s[0]) ** 2) > dist:
                        continue
                    # Now point is a neighbor
                    if val > threshold:
                        grow = False
            if grow:
                s[2] = dist
                growing = True
            else:
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        y = s[1] + dy
                        x = s[0] + dx
                        for i, line in enumerate(image):
                            if not grow or abs(i - y) > dist:
                                continue
                            for j, val in enumerate(line):
                                if not grow or abs(j - x) > dist:
                                    continue
                                if np.sqrt((i - y) ** 2 + (j - x) ** 2) > dist:
                                    continue
                                # Now point is a neighbor
                                if val > threshold:
                                    grow = False
                if grow:
                    s[:] = [x, y, dist]
                    growing = True
            if not grow:
                maxxed.append(s[:2])
        # Merge spheres
        deletables = set()
        for i, s in enumerate(spheres):
            for j, other in enumerate(spheres[i + 1:]):
                if i in deletables or j in deletables:
                    continue
                dist = np.sqrt((other[1] - s[1]) ** 2 + (other[0] - s[0]) ** 2)
                if s[2] < other[2]:
                    if dist <= other[2]:
                        deletables.add(i)
                elif other[2] < s[2]:
                    if dist <= s[2]:
                        deletables.add(i + j + 1)
                continue
                tests = (
                    (dist + other[2] <= s[2], i + j + 1),
                    (dist + s[2] <= other[2], i)
                )
                choice = np.random.randint(1)
                if tests[choice][0]:
                    deletables.add(tests[choice][1])
                elif tests[1 - choice][0]:
                    deletables.add(tests[1 - choice][1])
        for d in sorted(deletables, reverse=True):
            spheres.pop(d)
    # Last merge for spheres of same size
    n_spheres = -1
    while n_spheres != len(spheres):
        n_spheres = len(spheres)
        deletables = set()
        for i, s in enumerate(spheres):
            for j, other in enumerate(spheres[i + 1:]):
                if i in deletables or j in deletables:
                    continue
                dist = np.sqrt((other[1] - s[1]) ** 2 + (other[0] - s[0]) ** 2)
                if s[2] < other[2]:
                    if dist <= other[2]:
                        deletables.add(i)
                else:
                    if dist <= s[2]:
                        deletables.add(i + j + 1)
                continue
                tests = (
                    (dist + other[2] <= s[2], i + j + 1),
                    (dist + s[2] <= other[2], i)
                )
                choice = np.random.randint(1)
                if tests[choice][0]:
                    deletables.add(tests[choice][1])
                elif tests[1 - choice][0]:
                    deletables.add(tests[1 - choice][1])
        for d in sorted(deletables, reverse=True):
            spheres.pop(d)
    return spheres


def sphere_simplify(image):
    """
    Detect keypoints in image based on growing spheres

    Parameters
    ----------
    image : TYPE
        Input inmage formatted.

    Returns
    -------
    spheres : list[list[int, int, int]]
        Detected keypoints on the image.

    """
    # Apply gaussian blur
    kernel = [15, 15]
    image = cv2.GaussianBlur(image, kernel, 0)


    spheres = []
    threshold = 240
    for i, line in enumerate(image):
        for j, col in enumerate(line):
            if col < threshold and not i % 3 and not j % 3:
                spheres.append([j, i, 0])
    grow_spheres(spheres, image, threshold)

    if _DEBUG:
        # Draw spheres
        sphered = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for s in spheres:
            sphered = cv2.circle(
                sphered, s[:2], s[2],
                (s[2] % 3 * 255, (s[2] + 1) % 3 * 255, (s[2] + 2) % 3 * 255),
                1
            )
        cv2.imshow("Mini-spheres", sphered)
    # Normalize sphere sizes
    spheres = tuple((s[0] / PPK, s[1] / PPK, s[2] / PPK) for s in spheres)
    return spheres



def simplify_image(image):
    """
    Generate a set of keypoints usable by a computer for an image.

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # gradient_simplify(image)
    return sphere_simplify(image)


def get_kanjis():
    with open("data/handwritten.json") as database:
        data = json.load(database)
    return data


def match_to_kanji(input_vec, kanji, weights):
   #_, resis = np.linalg.lstsq(kanji, input_vec)[:2]
   total_dist = 0
   for pos in input_vec:
       dist = float('inf')
       for target in kanji:
           test_dist = np.sum((pos - target) ** 2)
           if test_dist < dist:
               dist = test_dist
       total_dist += dist
   return total_dist
   return np.linalg.norm(kanji - input_vec) # * weights)


def register_keypoints(keypoints, n_features, name=None):
    """Register new keypoints set for a kanji."""
    points = tuple(sorted(keypoints, key=lambda x: x[2], reverse=True))
    points = points[:n_features * 2]
    if _DEBUG:
        print(
            "Unormalized keypoints:",
            list(map(
                lambda x: (int(x[0] * PPK), int(x[1] * PPK), int(x[2] * PPK)),
                points
            ))
        )
    points = [list(x[:2]) for x in points]
    if name:
        print(',\n["{0}", {1}]'.format(name, points))
    else:
        print(points)
    return points


def get_kanji_candidates(keypoints, strokes):
    """Match keypoints against known kanji."""
    # No kanji is above 34 strokes, more or less 68 corners
    circles = tuple(sorted(keypoints, key=lambda x: x[2], reverse=True))
    radi = np.array([x[2] for x in circles])
    input_vec = np.array([x[:2] for x in circles])
    scores = []
    kanjis = get_kanjis()
    best = float('inf')
    for kanji in kanjis:
        for i in range(2, min(10, len(circles))):
            scores.append([
                kanji[0],
                match_to_kanji(
                    input_vec[:i + 1], kanji[1],
                    radi[:i + 1] / sum(radi[:i + 1])
                ),
                i]
            )
    scores = sorted(scores, key=lambda x: x[1])
    print(scores)


def main():
    """Main descriptor function."""
    args = parse_args()
    if _LEARNING:
        print("LEARNING")
    image = get_image(args.image)
    image = format_image(image)
    keypoints = simplify_image(image)
    if _LEARNING:
        register_keypoints(keypoints, int(args.strokes), args.name)
    else:
        get_kanji_candidates(keypoints, strokes=int(args.strokes))
    if _DEBUG:
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
