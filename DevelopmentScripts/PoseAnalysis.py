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


def extractCyclesByEuclidean(slidingWindowsDimension, keyPoints, videoname='', plotChart=False, sequence1=None):
    """
    Return the values in terms of frames that corresponds to the start and the end of cycles using the euclidean distance.
    The idea is to take a sliding window of a certain dimension (slidingWindowsDimension) that slides along the frames of
    the video and check the distance with the first window (the initial part of the video with dimensions
    slidingWindowDimension), then perform the euclidean and get, for each value of t (the shift value), the distance
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
        # perform the euclidean distance for each point of each frame and sum everything
        for i in range(0, slidingWindowsDimension):
            for j in range(0, keyPoints.shape[1]):
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


def checkExerciseByMeanInStdRange(trainerMeans, stds, userMeans, weights, path, errorStd):
    wrongPoses = []
    wrongPosesIndex = [0] * len(path)
    for couple in range(len(path)):
        trainerPose = path[couple][0]
        userPose = path[couple][1]
        error = 0
        keypointsWrong = []
        for keypoint in range(0, len(trainerMeans[trainerPose])):
            # define for each point the range of standard deviation of the trainer mean using a factor 3 as permitted
            # error coefficent and using the weights
            stdRangeXp = (trainerMeans[trainerPose][keypoint][0] + stds[trainerPose][keypoint][0])
            stdRangeXm = (trainerMeans[trainerPose][keypoint][0] - stds[trainerPose][keypoint][0])
            stdRangeYp = (trainerMeans[trainerPose][keypoint][1] + stds[trainerPose][keypoint][1])
            stdRangeYm = (trainerMeans[trainerPose][keypoint][1] - stds[trainerPose][keypoint][1])
            userMeanY = userMeans[userPose][keypoint][1]
            userMeanX = userMeans[userPose][keypoint][0]

            # control condition
            if (stdRangeXm > userMeanX or stdRangeXp < userMeanX) or (stdRangeYm > userMeanY or stdRangeYp < userMeanY):
                error += weights[keypoint][1]
                keypointsWrong.append(keypoint)
        if error > errorStd:
            wrongPoses.append([trainerPose, keypointsWrong, 'Error weight: ' + str(round(error, 3))])
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


def compareChecker(trainerMeans, userMeans, stds, path, weights, errorStd, errorAllowed=10):
    wrongPosesMeaninStd, wrongPosesMinSTDIndex = checkExerciseByMeanInStdRange(trainerMeans, stds,
                                                                               userMeans,
                                                                               weights, path, errorStd)
    # print('\nWrong poses by user mean in std deviation trainer range\n\n', wrongPosesMeaninStd)
    print('\nWrong poses by user mean in std deviation trainer range')
    if len(wrongPosesMeaninStd) < (len(userMeans) // 2):
        print("\nYou have done a great work, errors:", len(wrongPosesMeaninStd))
    else:
        print("\nYou sucks, errors:", len(wrongPosesMeaninStd))
    print('\nTotal Poses: ', len(path))
    print('\n')

    wrongPoses, wrongPosesIndex = checkByGramMatrix(path, trainerMeans, userMeans, 10)
    print('\nWrong poses by gram matrix checker')
    if len(wrongPoses) < (len(userMeans) // 2):
        print("\nYou have done a great work, errors:", len(wrongPoses))
    else:
        print("\nYou sucks, errors:", len(wrongPoses))
    print('\nTotal Poses: ', len(path))
    print('\n')

    return wrongPosesMinSTDIndex, wrongPosesIndex
