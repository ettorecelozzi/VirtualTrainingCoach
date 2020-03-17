import fastdtw as dtw
from DevelopmentScripts.Utility import serializeKeyPointsSequence
import myDTW as mydtw

def getDtwPath(frameSequence1, frameSequence2, weights=None):
    """
    Apply dtw to the frame sequences
    :param frameSequence1: keypoints numpy array
    :param frameSequence2: keypoints numpy array
    :param weights: weights of the exercise
    :return: the path that represent the sequences aligned
    """
    # frameSequence1 = serializeKeyPointsSequence(frameSequence1, weights)
    # frameSequence2 = serializeKeyPointsSequence(frameSequence2, weights)
    distance, path = mydtw.dtw(frameSequence1, frameSequence2)
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


def align1frame1pose(keyPoints, mins, weights=None):
    """
    Given the local mins of the cycles, align the cycles of an exercise execution
    :param keyPoints: numpy array of keypoints (#frames,25,2)
    :param mins: frame value that defines the cycles
    :param weights: weights of the joints
    :return: matrix of the poses aligned
    """
    i = 0
    paths = []
    while i in range(0, len(mins) - 2):
        # perform the dtw and get the path
        frameSequence1 = keyPoints[mins[i]:mins[i + 1]]
        frameSequence2 = keyPoints[mins[i + 1]: mins[i + 2]]
        path = getDtwPath(frameSequence1, frameSequence2, weights)

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

def align1frame1poseFirstCycle(keyPoints, mins, weights=None):
    """
    Given the local mins of the cycles, align the cycles of an exercise execution
    :param keyPoints: numpy array of keypoints (#frames,25,2)
    :param mins: frame value that defines the cycles
    :param weights: weights of the joints
    :return: matrix of the poses aligned
    """
    i = 0
    paths = []
    while i in range(0, len(mins) - 2):
        # perform the dtw and get the path
        frameSequence1 = keyPoints[mins[0]:mins[1]]
        frameSequence2 = keyPoints[mins[i + 1]: mins[i + 2]]
        path = getDtwPath(frameSequence1, frameSequence2, weights)

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
