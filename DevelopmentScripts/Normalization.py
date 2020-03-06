import numpy as np


def normalize(meanTorso, meanHipX, meanHipY, keypoints):
    """
    Keypoints are normalized using the distance from neck to the hip (torso)
    Each points is translated with respect of the meanHip that is aligned to the center (along both the axis).
    :param meanTorso: float
    :param meanHipX: float
    :param meanHipY: float
    :param keypoints: numpy array of shape (#frames, 25, 2)
    :return keypoints normalized
    """
    centerHeight = 720 // 2
    centerWidth = 1280 // 2
    # if meanHipX is at the left of centerWidth the translation value is centerWidth - meanHipX and is positive
    if meanHipX < centerWidth:
        translationX = centerWidth - meanHipX
    # if meanHipX is at the right of centerWidth the translation value is meanHipX - centerWidth and is negative
    # if exactly in the middle than the value is zero and no translation is performed
    else:
        translationX = -(meanHipX - centerWidth)
    # if meanHipY is at the left of centerHeigth the translation value is centerHeight - meanHipY and is positive
    if meanHipY < centerHeight:
        translationY = centerHeight - meanHipY
    # if meanHipY is at the right of centerHeigth the translation value is meanHipY - centerHeight and is negative
    # if exactly in the middle than the value is zero and no translation is performed
    else:
        translationY = -(meanHipY - centerHeight)

    for frame in keypoints:
        pointCounter = 0
        for points in frame:
            # for each point in each frame in keypoints we apply the normalization (if the point is not zero -> openpose error)
            if points[0] != 0 and points[1] != 0:
                points[0] += translationX
                points[1] += translationY
                points[0] = points[0] / meanTorso
                points[1] = points[1] / meanTorso
            pointCounter += 1
    return keypoints
