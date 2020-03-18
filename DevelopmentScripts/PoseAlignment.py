from DevelopmentScripts import myDTW


def getDtwPath(frameSequence1, frameSequence2, weights=None):
    """
    Apply dtw to the frame sequences
    :param frameSequence1: keypoints numpy array
    :param frameSequence2: keypoints numpy array
    :param weights: weights of the exercise
    :return: the path that represent the sequences aligned
    """
    distance, path = myDTW.dtw(frameSequence1, frameSequence2)
    return path


def findNext(value, paths, pathsIndex):
    """
    Recursive function to find all the frames that belongs to the same pose in different cycles.
    e.g. 0|0 is connected with 0|1; so the function search all the frames in other cycles where there is 0|1. So if
    there is a couple 0|1,0|2 then 0|2 will be added the the list of the same poses.
    :param value: element we are searching e.g. 0|1 -> we search for elements that start for 0|1
    :param paths:matrix with all the paths
    :param pathsIndex: index in the paths matrix (tells what cycle in the paths we are reffering to
    :return:
    """
    i = 0
    j = 0
    tmp = []
    # the while check if we should go on searching for element in the list
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


def align1frame1pose_tmp(keyPoints, mins, weights=None):
    """
    Given the local mins of the cycles, align the cycles of an exercise execution
    :param keyPoints: numpy array of keypoints (#frames,25,2)
    :param mins: frame value that defines the cycles
    :param weights: weights of the joints
    :return: matrix of the poses aligned
    """
    i = 0
    paths = []
    bound = len(mins) - 2 if len(mins) > 3 else len(mins) - 1
    while i in range(0, bound):
        # perform the dtw and get the path
        frameSequence1 = keyPoints[mins[i]:mins[i + 1]]
        frameSequence2 = keyPoints[mins[i + 1]: mins[i + 2]] if len(mins) > 3 else keyPoints[mins[i + 1]:]
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
    bound = len(mins) - 2 if len(mins) > 3 else len(mins) - 1
    while i in range(0, bound):
        # perform the dtw and get the path
        frameSequence1 = keyPoints[mins[i]:mins[i + 1]]
        frameSequence2 = keyPoints[mins[i + 1]: mins[i + 2]] if len(mins) > 3 else keyPoints[mins[i + 1]:]
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
            pose += findNextFirstCycle(str(paths[0][i][j][1]) + '|0', paths, 1)
        pose = list(dict.fromkeys(pose))
        poseMatrix.append(pose)
    return poseMatrix


def findNextFirstCycle(value, paths, pathsIndex):
    """
    Recursive function to find all the frames that belongs to the same pose in different cycles.
    :param value: element we are searching e.g. 0|1 -> we search for elements that start for 0|1
    :param paths:matrix with all the paths
    :param pathsIndex: index in the paths matrix (tells what cycle in the paths we are reffering to
    :return:
    """
    i = 0
    j = 0
    tmp = []
    # the while check if we should go on searching for element in the list
    while paths[pathsIndex][i][j][0] <= int(value.split('|')[0]) and i < len(paths[pathsIndex]) - 1:
        alreadySeen = False
        for j in range(0, len(paths[pathsIndex][i])):
            if int(value.split('|')[0]) == paths[pathsIndex][i][j][0]:
                if alreadySeen is False:
                    tmp += [str(paths[pathsIndex][i][0][0]) + '|' + str(pathsIndex)]
                    alreadySeen = True
                if pathsIndex < len(paths) - 1:
                    tmp += findNextFirstCycle(value, paths, pathsIndex + 1)
        i += 1
        j = 0
    return tmp
