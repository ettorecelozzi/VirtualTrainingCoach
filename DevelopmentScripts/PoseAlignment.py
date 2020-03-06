import fastdtw as dtw
from DevelopmentScripts import PoseAnalysis
import numpy as np
import DevelopmentScripts.Statistics as stat


def getDtwPath(frameSequence1, frameSequence2, weights=None, angles=False):
    """
    Apply dtw to the frame sequences
    :param frameSequence1: keypoints numpy array
    :param frameSequence2: keypoints numpy array
    :param weights: weights of the exercise
    :param angles: boolean to specify if the angles are considered
    :return: the path that represent the sequences aligned
    """
    if angles is False:
        frameSequence1 = PoseAnalysis.serializeKeyPointsSequence(frameSequence1, weights)
        frameSequence2 = PoseAnalysis.serializeKeyPointsSequence(frameSequence2, weights)
    distance, path = dtw.dtw(frameSequence1, frameSequence2)
    return path


def findNext(value, paths, pathsIndex):
    """

    :param value:
    :param paths:
    :param pathsIndex:
    :return:
    """
    i = 0
    j = 0
    tmp = []
    while paths[pathsIndex][i][j][0] <= int(value.split('|')[0]) and i < len(paths[pathsIndex]) - 1:
        alreadySeen = False
        for j in range(0, len(paths[pathsIndex][i])):
            if int(value.split('|')[0]) == paths[pathsIndex][i][j][0]:
                if alreadySeen is False:
                    tmp += [str(paths[pathsIndex][i][0][0]) + '|' + str(pathsIndex)]
                    alreadySeen = True
                if pathsIndex < len(paths) - 1:
                    tmp += findNext(str(paths[pathsIndex][i][j][1]) + '|' + str(pathsIndex + 1), paths, pathsIndex + 1)
        i += 1
        j = 0
    return tmp


def align1frame1pose(keyPoints, mins, weights=None, angles=False):
    """
    Given the local mins of the cycles, align the cycles of an exercise execution
    :param keyPoints: numpy array of keypoints (#frames,25,2)
    :param mins: frame value that defines the cycles
    :param weights: weights of the joints
    :param angles: boolean to specify if the angles are used
    :return: ???
    """
    i = 0
    paths = []
    while i in range(0, len(mins) - 2):
        # perform the dtw and get the path
        frameSequence1 = keyPoints[mins[i]:mins[i + 1]]
        frameSequence2 = keyPoints[mins[i + 1] + 1: mins[i + 2]]
        path = getDtwPath(frameSequence1, frameSequence2, weights, angles=angles)

        unifiedPath = []
        tmp = [path[0]]
        for j in range(1, len(path)):
            if path[j - 1][0] == path[j][0]:
                tmp.append(path[j])
            else:
                unifiedPath.append(tmp)
                tmp = [path[j]]
        unifiedPath.append(tmp)
        paths.append(unifiedPath)
        i += 1
    poseMatrix = []
    for i in range(0, len(paths[0])):
        pose = []
        for j in range(0, len(paths[0][i])):
            pose += [str(paths[0][i][0][0]) + '|0']
            pose += findNext(str(paths[0][i][j][1]) + '|1', paths, 1)
        pose = list(dict.fromkeys(pose))
        poseMatrix.append(pose)
    return poseMatrix


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
    From pose's alignedList we remove the duplicates to avoid that in the means and std deviation calculation, a value is
    considered more than one time. Once removed we retrieve the means and the std deviation for each pose.
    :param alignedLists: list. aligned poses
    :param mins: list. bound of the cycles
    :param keyPoints: array
    :return: arrays: mean, stds
    """
    means = np.array([])
    stds = np.array([])
    for poseToStat in alignedLists:
        noDuplicateList = removeDuplicate(poseToStat)
        mean, std = stat.getStatistics(noDuplicateList, mins, keyPoints, 25)
        means = np.append(means, mean)  # the list of the means, one for each pose, each long 25
        stds = np.append(stds, std)  # the list of the stds deviations, one for each pose, each long 25
    return means, stds
