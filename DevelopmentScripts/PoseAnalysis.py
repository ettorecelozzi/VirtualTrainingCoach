import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy import linalg as la
from sklearn.cluster import AgglomerativeClustering
from DevelopmentScripts import myDTW


def findmin(radius, x, y, title, plotChart):
    """
    Find the local minus values of a vector y using the radius to separate the mins by at least radius values and plot the
    chart
    :param radius: radius to check for local minima
    :param x: frames
    :param y: distances
    :param title: string
    :param plotChart: boolean
    :return: list of bounds of cycles (list of int)
    """
    percentile = np.percentile(y, 7)

    ytominimize = np.array([])
    # we need to add a huge value in the first position in order to make the first value a localMin (the left(inf) and the
    # right value are both greater
    ytominimize = np.append(ytominimize, np.inf)
    ytominimize = np.append(ytominimize, y)
    # argrelextrema finds the local mins in a vector (ytominimize) separated by at least radious values
    localMins = argrelextrema(ytominimize, np.less, order=radius)[0]
    realMins = []

    meanMins = np.mean(y[localMins])
    threshold = meanMins + (percentile)  # - min(y[localMins]))
    mins = np.sort(np.asarray([y[i] for i in localMins]))
    dist = [threshold - d for d in mins[:len(mins) - 2]]
    meanDist = np.mean(dist)
    testx = np.array([])
    testy = np.array([])
    for i in localMins:
        if y[i - 1] < meanDist:  # i-1 becuase I added an element in ytominimize
            realMins.append(i - 1)
            testx = np.append(testx, i - 1)
            testy = np.append(testy, y[i - 1])
    realMins = np.sort(realMins)
    # Plot the chart if plotChart is True
    if plotChart:
        plt.scatter(x, y, color="red")
        plt.scatter(testx, testy, color="blue")
        threshold = [threshold] * len(x)
        meanDist = [meanDist] * len(x)
        plt.plot(range(len(x)), threshold, 'blue')
        plt.plot(range(len(x)), meanDist, 'green')
        plt.title(title)
        plt.show()
    return realMins


def findminClustering(radius, x, y, title, plotChart):
    """
    Find to local minimas through a hierarchical clustering algorithm
    :param radius: radius to check for local minima
    :param x: frames
    :param y: distances
    :param title: string
    :param plotChart: boolean
    :return: list of bounds of cycles (list of int)
    """
    ytominimize = np.array([])
    ytominimize = np.append(ytominimize, np.inf)
    ytominimize = np.append(ytominimize, y)
    localMins = argrelextrema(ytominimize, np.less, order=radius)[0]
    mins = np.asarray([y[i] for i in localMins])
    mins = np.reshape(mins, newshape=(mins.shape[0], 1))
    mins = np.insert(mins, 0, values=0, axis=1)

    # Ward variance minimization algorithm used to calculate the distance between the new formed clusters
    cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    cluster.fit_predict(mins)
    realMins = [m for l, m in zip(cluster.labels_, localMins) if l == 0]
    miny = [y[i] for i in realMins]
    if plotChart:
        plt.scatter(x, y, color="red")
        plt.scatter(realMins, miny, color="blue")
        plt.title(title)
        plt.show()
    return realMins


def extractCyclesByDtw(slidingWindowsDimension, keyPoints, plotChart=False, sequence1=None):
    """
    Return the values in terms of frames that corresponds to the start and the end of cycles using the dtw as a distance
    measure.
    The idea is to take a sliding window of a certain dimension (slidingWindowsDimension) that slides along the frames of
    the video and check the distance with the first window (the initial part of the video with dimensions slidingWindowDimesnion),
    then perform the dtw and get, for each value of t (the shift value), the distance
    :param slidingWindowsDimension: int
    :param keyPoints: array
    :param plotChart: boolean to decide if plot the chart of the cycle extraction
    :param sequence1: array. Used to extract cycle of the User
    :param user: boolean. User case
    :return: bound of the cycles
    """
    t = 0
    # take the first window
    if sequence1 is None:
        sequence1 = keyPoints[t:t + slidingWindowsDimension]

    x = np.array([])
    y = np.array([])
    # slide the sliding window and perform the dtw distance
    for t in range(t, keyPoints.shape[0] - slidingWindowsDimension):
        sequence2 = keyPoints[t:t + slidingWindowsDimension]
        distance, path = myDTW.dtw(sequence1, sequence2)
        x = np.append(x, t)
        y = np.append(y, distance)

    return findminClustering(slidingWindowsDimension // 2, x, y, 'Dtw', plotChart)


def extractCyclesByEuclidean(slidingWindowsDimension, keyPoints, videoname='', weights=None, plotChart=False,
                             sequence1=None):
    """
    Return the values in terms of frames that corresponds to the start and the end of cycles using the euclidean distance.
    The idea is to take a sliding window of a certain dimension (slidingWindowsDimension) that slides along the frames of
    the video and check the distance with the first window (the initial part of the video with dimensions
    slidingWindowDimension), then perform the euclidean and get, for each value of t (the shift value), the distance
    :param slidingWindowsDimension: int
    :param keyPoints: array of keypoints
    :param videoname: string
    :param weights: list of weights for each joint
    :param plotChart: boolean to decide if plot the chart of the cycle extraction
    :param sequence1: array. Used to extract cycle of the User
    :return: bound of the cycles
    """
    t = 0
    # take the first window
    if sequence1 is None:
        sequence1 = keyPoints[t:t + slidingWindowsDimension]

    x = np.array([])
    y = np.array([])
    # slide the sliding window
    for t in range(t, keyPoints.shape[0] - slidingWindowsDimension):
        sequence2 = keyPoints[t:t + slidingWindowsDimension]
        distance = 0
        # perform the euclidean distance for each point of each frame and sum everything
        for i in range(0, slidingWindowsDimension):
            for j in range(0, keyPoints.shape[1]):
                if weights is None or (weights is not None and weights[j][1] != 0):
                    v = np.power(float(sequence1[i][j][0]) - float(sequence2[i][j][0]), 2) + np.power(
                        float(sequence1[i][j][1]) - float(sequence2[i][j][1]), 2)
                    distance = distance + np.sqrt(v)
        x = np.append(x, t)
        y = np.append(y, distance)

    return findmin(slidingWindowsDimension // 2, x, y, 'Euclidean - ' + videoname, plotChart)


def extractCyclesByGram(slidingWindowsDimension, keyPoints, plotChart=False, sequence1=None):
    """
    Return the values in terms of frames that corresponds to the start and the end of cycles using the gram distance.
    The idea is to take a sliding window of a certain dimension (slidingWindowsDimension) that slides along the frames of
    the video and check the distance with the first window (the initial part of the video with dimensions slidingWindowDimesnion),
    then perform the gram and get, for each value of t (the shift value), the distance
    :param slidingWindowsDimension: int
    :param keyPoints: array of keypoints
    :param plotChart: boolean to decide if plot the chart of the cycle extraction
    :param sequence1: array. Used to extract cycle of the User
    :return: bound of the cycles
    """
    t = 0
    # take the first window
    if sequence1 is None:
        sequence1 = keyPoints[t:t + slidingWindowsDimension]

    x = np.array([])
    y = np.array([])
    # slide the sliding window
    for t in range(t, keyPoints.shape[0] - slidingWindowsDimension):
        sequence2 = keyPoints[t:t + slidingWindowsDimension]
        distance = 0
        # perform the gram distance for each frame and sum everything
        for i in range(0, slidingWindowsDimension):
            distance = distance + getGramDistance(sequence1[i], sequence2[i])
        print("t: " + str(t) + " dist: " + str(distance))
        x = np.append(x, t)
        y = np.append(y, distance)

    return findmin(slidingWindowsDimension // 2, x, y, 'Gram', plotChart)


def getMeanMeasures(keyPointsSequence, meanRange):
    """
    Return the mean measures in a neighborhood(meanRange) of the central frame of the torso distance
    (for scale normalization), midhip x and midhip y (for traslation normalization)
    :param keyPointsSequence: keypoints array (#frames, 25, 2)
    :param meanRange: int. range to consider for the mean
    :return: mean torso, mean midHipx, mean midHipy
    """
    # for each frame in the cycle take the euclidean distance of the torso
    torsoMeasures = np.array([])
    midhipxMeasures = np.array([])
    midhipyMeasures = np.array([])

    # take all the keyPoint in a neighborhood(meanRange) of the central frame
    for keyPointsFrame in keyPointsSequence[
                          len(keyPointsSequence) // 2 - meanRange:len(keyPointsSequence) // 2 + meanRange]:
        neck = np.array([float(keyPointsFrame[1][0]), float(keyPointsFrame[1][1])])
        midhip = np.array([float(keyPointsFrame[8][0]), float(keyPointsFrame[8][1])])
        # append the midhip x and y measures to their lists
        midhipxMeasures = np.append(midhipxMeasures, float(keyPointsFrame[8][0]))
        midhipyMeasures = np.append(midhipyMeasures, float(keyPointsFrame[8][1]))
        # append the distance from neck to midhip
        torsoMeasures = np.append(torsoMeasures, euclidean(neck, midhip))
    # return the means
    return np.mean(torsoMeasures), np.mean(midhipxMeasures), np.mean(midhipyMeasures)


def checkExerciseByMeanInStdRange(trainerCycle, stdTrainer, userCycle, weights, path, errorStd):
    """
    Function that checks how the user perform the exercise comparing the standard deviation range of the keypoints
    :param trainerCycle: trainer poses
    :param stdTrainer: trainer std dev poses
    :param userCycle: user poses
    :param weights: weights of the joints, pass None to not use weights
    :param path: aligned path returned by the DTW
    :param errorStd: the error allowed to classify the exercise as correct
    :return: 2 arrays, the wrong poses and the indexes of the wrong couple
    """
    wrongPoses = []
    wrongPosesIndex = [0] * len(path)
    for couple in range(len(path)):
        trainerPose = path[couple][0]
        userPose = path[couple][1]
        error = 0
        keypointsWrong = []
        for keypoint in range(0, len(trainerCycle[trainerPose])):
            if weights[keypoint][1] == 0: continue

            stdRangeXp = trainerCycle[trainerPose][keypoint][0] + stdTrainer[trainerPose][keypoint][0] + errorStd
            stdRangeXm = trainerCycle[trainerPose][keypoint][0] - stdTrainer[trainerPose][keypoint][0] - errorStd
            stdRangeYp = trainerCycle[trainerPose][keypoint][1] + stdTrainer[trainerPose][keypoint][1] + errorStd
            stdRangeYm = trainerCycle[trainerPose][keypoint][1] - stdTrainer[trainerPose][keypoint][1] - errorStd
            userMeanY = userCycle[userPose][keypoint][1]
            userMeanX = userCycle[userPose][keypoint][0]

            # control condition
            if (stdRangeXm > userMeanX or stdRangeXp < userMeanX) or (stdRangeYm > userMeanY or stdRangeYp < userMeanY):
                error += weights[keypoint][1]
                keypointsWrong.append(keypoint)
        if error > 0:
            wrongPoses.append([keypointsWrong, 'Error weight: ' + str(round(error, 3))])
            wrongPosesIndex[couple] = -1
    return wrongPoses, wrongPosesIndex


def getGramDistance(poseDescriptor1, poseDescriptor2):
    """
    Perform gram distance between two poses
    :param poseDescriptor1: first pose
    :param poseDescriptor2: second pose
    :return: distance between the two poses
    """
    g1 = np.outer(poseDescriptor1, poseDescriptor1.T)
    g2 = np.outer(poseDescriptor2, poseDescriptor2.T)
    distance = np.trace(g1) + np.trace(g2) - 2.0 * np.trace(
        la.fractional_matrix_power(np.dot(la.fractional_matrix_power(g1, 0.5),
                                          np.dot(g2, la.fractional_matrix_power(g1, 0.5))), 0.5))
    return distance


def checkByGramMatrix(path, trainerMeans, userMeans, distance_error):
    """
    Check validity of an exercise using the gram distance
    :param path: dtw path of same poses between trainer and user
    :param trainerMeans: mean of the trainer poses across all cycles
    :param userMeans: mean of the user poses across all cycles
    :param distance_error: error allowed
    :return: the list of the wrong poses (error greater than the allowed error)
    """
    posesWrong = []
    posesWrongIndex = [0] * len(path)
    for couple in range(len(path)):
        trainerPose = path[couple][0]
        userPose = path[couple][1]
        pose_descriptor_trainer = trainerMeans[trainerPose]
        pose_descriptor_user = userMeans[userPose]
        distance = getGramDistance(pose_descriptor_trainer, pose_descriptor_user)
        if distance > distance_error:
            posesWrong.append([trainerPose, userPose])  # number of errors is equal to the length of the poseWrong list
            posesWrongIndex[couple] = -1
    return posesWrong, posesWrongIndex


def getPointsAngles(means, weights):
    """
    For each frame retrieve the angle between the joints specified in the 'jointsNumber' list.
    To get cosine of the angle the dot product formula is used, then the arccos returns the angle
    :param means: array of the mean poses
    :param weights: weights of the joints
    :return: array of the joints angles
    """
    angles = []
    for pose in means:
        frameAngles = getPoseAngle(pose,weights)
        angles.append(frameAngles)
    return angles

def getPoseAngle(pose, weights):
    """
    For a pose retrieve the angle between the joints specified in the 'jointsNumber' list.
    To get cosine of the angle the dot product formula is used, then the arccos returns the angle
    :param pose: pose
    :param weights: weights of the joints
    :return: array of the joints angles
    """
    jointsNumber = {'leftUpperBody': [[8, 1], [1, 2], [2, 3], [3, 4]],
                    'rightUpperBody': [[8, 1], [1, 5], [5, 6], [6, 7]],
                    'leftLowerBody': [[1, 8], [8, 9], [9, 10], [10, 11], [11, 23]],
                    'rightLowerBody': [[1, 8], [8, 12], [12, 13], [13, 14], [14, 20]]}

    frameAngles = []
    for joints in jointsNumber.values():  # number of keypoints reduced
        for couple in range(len(joints) - 1):
            jointIndex = joints[couple]  # couple of joints that forms the vector
            couple += 1
            jointIndexN = joints[couple]  # next couple of joints that forms the vector
            if weights[jointIndex[1]][1] == 0: continue
            bodyVector1 = np.subtract(np.array([float(pose[jointIndex[1]][0]), float(pose[jointIndex[1]][1])]),
                                      np.array([float(pose[jointIndex[0]][0]), float(pose[jointIndex[0]][1])]))
            bodyVector2 = np.subtract(np.array([float(pose[jointIndexN[0]][0]), float(pose[jointIndexN[0]][1])]),
                                      np.array([float(pose[jointIndexN[1]][0]), float(pose[jointIndexN[1]][1])]))
            # vector normalization
            bodyVector1 = np.divide(bodyVector1, np.linalg.norm(bodyVector1))
            bodyVector2 = np.divide(bodyVector2, np.linalg.norm(bodyVector2))
            # "inverse" dot product
            angle = np.degrees(np.arccos(np.clip((np.dot(bodyVector1, bodyVector2)), -1.0, 1.0)))
            # string to identify the calculated angle
            strAngle = str(jointIndex[1]) + "-" + str(jointIndex[0]) + "|" + \
                       str(jointIndexN[0]) + "-" + str(jointIndexN[1])
            # list of: joint linked to neck, angle formed with torso, each long 25
            frameAngles.append([strAngle, angle])
    return frameAngles

def checkByJointsAngles(trainerCycle, userCycle, weights, path, errorAngles):
    """
    Function that checks how the user perform the exercise comparing the trainer angles between the joints with the user
    ones.
    :param trainerCycle: trainer poses
    :param userCycle: user poses
    :param weights: weights of the joints, pass None to not use weights
    :param path: aligned path returned by the DTW
    :param errorAngles: the error allowed to classify the exercise as correct
    :return: 2 arrays, the wrong poses and the indexes of the wrong couple
    """
    trainerAngles = getPointsAngles(trainerCycle, weights)
    userAngles = getPointsAngles(userCycle, weights)
    wrongPoses = []
    wrongPosesIndex = [0] * len(path)
    for couple in range(len(path)):
        trainerPose = path[couple][0]
        userPose = path[couple][1]
        error = 0
        keypointsWrong = []
        for keypoint in range(len(trainerAngles[trainerPose])):
            # note that at this point the keypoint 0 is the first angle calculated not the keypoint number 0
            trainerAngle = trainerAngles[trainerPose][keypoint][1]
            userAngle = userAngles[userPose][keypoint][1]
            if userAngle > trainerAngle + errorAngles or userAngle < trainerAngle - errorAngles:
                error += weights[int(trainerAngles[trainerPose][keypoint][0][0])][1]
                keypointsWrong.append(trainerAngles[trainerPose][keypoint][0])
        if error > 0:
            wrongPoses.append([keypointsWrong, 'Error weight: ' + str(round(error, 3))])
            wrongPosesIndex[couple] = - 1
    return wrongPoses, wrongPosesIndex

def getAngleDistance(trainerPose,userPose,errorAngles,weights):
    trainerAngles = getPoseAngle(trainerPose,weights)
    userAngles = getPoseAngle(userPose, weights)
    error = 0
    keypointsWrong = []
    for keypoint in range(len(trainerAngles)):
        # note that at this point the keypoint 0 is the first angle calculated not the keypoint number 0
        trainerAngle = trainerAngles[keypoint][1]
        userAngle = userAngles[keypoint][1]
        if userAngle > trainerAngle + errorAngles or userAngle < trainerAngle - errorAngles:
            error += weights[int(trainerAngles[keypoint][0][0])][1] * abs(trainerAngle-userAngle)
    return error

def compareChecker(trainerCycle, userCycle, path, weights, errorAngles):
    """
    Find out the wrong poses performed by the user
    :param trainerCycle: trainer poses
    :param userCycle: user poses
    :param stdsTrainer: trainer standard deviation
    :param path: aligned path returned by the DTW
    :param weights: weights of the joints, pass None to not use weights
    :param errorStd: the error allowed to classify the exercise as correct for the std checker
    :param errorAngles: the error allowed to classify the exercise as correct for the angles checker
    """
    # wrongPoses, wrongPosesIndex = checkExerciseByMeanInStdRange(trainerCycle=trainerCycle, stdTrainer=stdsTrainer,
    #                                                             userCycle=userCycle, path=path, weights=weights,
    #                                                             errorStd=errorStd)
    # print('\nWrong poses STD:')
    # print(wrongPoses)
    # print('Couple of poses wrong STD:')
    # notZero = np.nonzero(wrongPosesIndex)[0]
    # for i in notZero: print(str(path[i]), end=', ')
    # print('\nNumber of errors STD: ' + str(len(notZero)) + '/' + str(len(path)))

    wrongPosesAngles, wrongPosesIndexAngles = checkByJointsAngles(trainerCycle=trainerCycle, userCycle=userCycle,
                                                                  path=path, weights=weights, errorAngles=errorAngles)
    print('\nWrong joints ANGLES:')
    print(wrongPosesAngles)
    print('Couple of poses wrong ANGLES:')
    notZeroAngles = np.nonzero(wrongPosesIndexAngles)[0]
    wrong = []
    for i in notZeroAngles:
        wrong.append(path[i])
        print(str(path[i]), end=', ')
    print('\nNumber of errors ANGLES: ' + str(len(notZeroAngles)) + '/' + str(len(path)))
    return wrong
