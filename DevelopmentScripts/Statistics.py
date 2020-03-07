import numpy as np


def getStatistics(alignedList, mins, keyPoints):
    """
    Mean and standard deviation retrieved from the aligned poses
    :param alignedList: alignedList of a pose
    :param mins: array. Mins extracted
    :param keyPoints: array of keypoints
    :param bodyKeypoints: int. number of joints
    :return: arrays. Mean and standard deviation
    """
    framePoints = []
    for couple in alignedList:
        frame, cycle = [int(i) for i in couple.split('|')]
        if cycle != len(mins) - 1:
            pointsCycle = keyPoints[mins[cycle]:mins[cycle + 1]]
            framePoints.append(pointsCycle[frame])  # list append is faster than numpy append
        else:
            pointsCycle = keyPoints[mins[cycle]:]
            framePoints.append(pointsCycle[frame])
    framePoints = np.asarray(framePoints)  # numpy arrays are handy and memory efficient
    mean = np.mean(framePoints, axis=0)
    stdDev = np.std(framePoints, axis=0)
    return mean, stdDev


def removeDuplicate(unifiedPaths):
    """
    The function delete the duplicated elements from unifiedPaths list and return a free-duplicated elements list
    for ex. [ [ (0|1), (0|2) ], [ (0|2), (1|3) ] ] results in [ 0|1, 0|2 , 1|3]
    :param unifiedPaths: list to clean
    :return: cleaned list
    """
    unifiedSet = []
    for subset in unifiedPaths:
        # part for the pose plot
        if type(subset) != list:
            if subset not in unifiedSet:
                unifiedSet.append(subset)
                continue
        # part of the precedent align
        for tuple in subset:
            if tuple[0] not in unifiedSet:
                unifiedSet.append(tuple[0])
            if tuple[1] not in unifiedSet:
                unifiedSet.append(tuple[1])
    return unifiedSet


def removeDuplicateAndGetStat(alignedLists, mins, keyPoints):
    """
    From pose's alignedList the duplicates are removed to avoid that in the means and std deviation calculation, a value
    is considered more than one time. Once removed means and the std deviation are retrieved for each pose.
    :param alignedLists: list. aligned poses
    :param mins: list. bound of the cycles
    :param keyPoints: array
    :return: arrays: mean, stds
    """
    means = []
    stds = []
    for poseToStat in alignedLists:
        noDuplicateList = removeDuplicate(poseToStat)
        mean, std = getStatistics(noDuplicateList, mins, keyPoints)
        means.append(mean)  # the list of the means, one for each pose, each long 25
        stds.append(std)  # the list of the stds deviations, one for each pose, each long 25
    return np.asarray(means), np.asarray(stds)


def getStatAnglesPose(alignedList, mins, angles):
    """
    Mean and standard deviation retrieved from the aligned pose
    :param alignedList: alignedList of a pose
    :param angles: list of the joints' angles
    :param mins: array. Mins extracted
    :return: arrays. Mean and standard deviation
    :return: mean, standard deviation of angles
    """
    framePoints = []
    for couple in alignedList:
        frame, cycle = [int(i) for i in couple.split('|')]
        if cycle != len(mins) - 1:
            pointsCycle = angles[mins[cycle]:mins[cycle + 1]]
            framePoints.append(pointsCycle[frame])  # list append is faster than numpy append
        else:
            pointsCycle = angles[mins[cycle]:]
            framePoints.append(pointsCycle[frame])
    framePoints = np.asarray(framePoints)  # numpy arrays are handy and memory efficient
    mean = np.mean(framePoints, axis=0)
    stdDev = np.std(framePoints, axis=0)
    return mean, stdDev


def getStatisticAngles(alignedLists, mins, anglesKeypoints):
    """
    The means and the std deviation are retrieved for each pose.
    :param alignedLists: list. aligned poses
    :param mins: list. bound of the cycles
    :param anglesKeypoints: array
    :return: arrays: mean, stds
    """
    means = []
    stds = []
    for poseToStat in alignedLists:
        mean, std = getStatAnglesPose(poseToStat, mins, anglesKeypoints)
        means.append(mean)
        stds.append(std)
    return np.asarray(means), np.asarray(stds)
