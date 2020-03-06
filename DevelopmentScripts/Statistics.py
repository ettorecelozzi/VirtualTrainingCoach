import numpy as np


def getStatistics(alignedList, mins, keyPoints, bodyKeypoints):
    """
    Mean and standard deviation retrieved from the aligned poses
    :param alignedList: alignedList of a pose
    :param mins: array. Mins extracted
    :param keyPoints: array of keypoints
    :param bodyKeypoints: int. number of joints
    :return: arrays. Mean and standard deviation
    """
    # framePoints = np.full((bodyKeypoints, 2), -1)  # list filled with 25 "-1" to exploit the vertical concatenation
    framePoints = np.array([])
    # for each tuple in unifiedSet retrieve the frame and the cycle to get the keypoints and (vertical) concatenate each
    # frame's keypoints to calculate mean and std deviation
    for couple in alignedList:
        frame, cycle = [int(i) for i in couple.split('|')]
        if cycle != len(mins) - 1:
            pointsCycle = keyPoints[mins[cycle]:mins[cycle + 1]]
            framePoints = np.vstack((framePoints, pointsCycle[frame]))
        else:
            pointsCycle = keyPoints[mins[cycle]:]
            framePoints = np.vstack((framePoints, pointsCycle[frame]))
    mean = np.mean(framePoints, axis=0)
    stdDev = np.std(framePoints, axis=0)
    # mean = np.mean(framePoints[26:], axis=0)  # to remove the first 25 "-1" inserted
    # stdDev = np.std(framePoints[26:], axis=0)
    return mean, stdDev


def getStatisticsForAngles(alignedList, mins, angles):
    """
    Mean and standard deviation retrieved from the aligned poses
    :param alignedList: alignedList of a pose
    :param angles: list of the joints' angles
    :param mins: array. Mins extracted
    :return: arrays. Mean and standard deviation
    :return:
    """
    framePoints = np.array([])  # maybe same stragey of above is needed
    for couple in alignedList:
        frame, cycle = [int(i) for i in couple.split('|')]
        if cycle != len(mins) - 1:
            pointsCycle = angles[mins[cycle]:mins[cycle + 1]]
            framePoints += pointsCycle[frame]
        else:
            pointsCycle = angles[mins[cycle]:]
            framePoints = np.vstack((framePoints, pointsCycle[frame]))
    mean = np.mean(framePoints, axis=0)
    stdDev = np.std(framePoints, axis=0)
    return mean, stdDev
