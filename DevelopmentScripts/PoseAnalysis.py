import fastdtw as dtw
import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy import linalg as la


def serializeKeyPointsSequence(keyPointsSequence, weights=None):
    """
    The DTW is between two time sequences of points, but we have sequences of frames ( that are lists of points ) and we
    can't do the DTW between sequnces of sequences of points. The sequences of frame are serialized, meaning that
    for each frame we decouple each (x,y) so that a frame that at the beginning is [(x1,y1)(x2,y2)...] will became
    [x1,y1,x2,y2,...]
    :param keyPointsSequence: keypoints numpy array
    :param weights: weights of the joints
    :return: keypoints serialized
    """
    res = []
    for keyPointsFrame in keyPointsSequence:
        framePoints = []
        pointIndex = 0
        for point in keyPointsFrame:
            if weights is not None:
                if weights[pointIndex][1] != 0.0:
                    framePoints.append(point[0])
                    framePoints.append(point[1])
            else:
                framePoints.append(point[0])
                framePoints.append(point[1])

            pointIndex += 1
        res.append(framePoints)
    res = np.array(res)
    return res


def findmin(radius, x, y, title, plotChart, slidingWindowsDimension, user):
    """
    TODO: verify (not modified)
    Find the local minus values of a vector y using the radius to separate the mins by at least radius values and plot the
    chart
    :param radius:
    :param x:
    :param y:
    :param title:
    :param plotChart:
    :param slidingWindowsDimension:
    :param user:
    :return: list of bounds of cycles (list of int)
    """
    ytominimize = np.array([])
    # we need to add a huge value in the first position in order to make the first value a localMin (the left(inf) and the
    # right value are both greater
    ytominimize = np.append(ytominimize, np.inf)
    ytominimize = np.append(ytominimize, y)
    # argrelextrema finds the local mins in a vector (ytominimize) separated by at least radious values
    localMins = argrelextrema(ytominimize, np.less, order=radius)
    realMins = []
    # meany is the value under that the mins found are considered good and the value is taken doing the mean of all the
    # localMins found
    meany = np.mean(y[localMins[0][1:]])
    testx = np.array([])
    testy = np.array([])
    for i in localMins[0]:
        if y[i - 1] < meany + 4:  # i-1 becuase I added an element in ytominimize
            realMins.append(i - 1)
            testx = np.append(testx, i - 1)
            testy = np.append(testy, y[i - 1])
    # Plot the chart if plotChart is True
    if plotChart:
        plt.scatter(x, y, color="red")
        plt.scatter(testx, testy, color="blue")
        meany = [meany] * len(testx)
        plt.plot(testx, meany, 'blue')
        plt.title(title)
        plt.show()
    if (realMins[1] - realMins[0]) > slidingWindowsDimension * 3 and user is True:
        realMins.pop(0)
    return realMins


def extractCyclesByDtw(slidingWindowsDimension, keyPoints, plotChart=False, sequence1=None, user=False):
    """
    Return the values in terms of frames that corresponds to the start and the end of cycles using the dtw as a distance
    measure.
    The idea is to take a sliding window of a certain dimension (slidingWindowsDimension) that slides along the frames of
    the video and check the distance with the first window (the initial part of the video with dimensions slidingWindowDimesnion),
    then perform the dtw and get, for each value of t (the shift value), the distance
    :param slidingWindowsDimension: int
    :param keyPoints: array
    :param plotChart: boolean to decide if plot the chart of the cycle extraction
    :param sequence1: array. Used to extract cycle of the user
    :param user: boolean. User case
    :return: bound of the cycles
    """
    t = 0
    # take the first window
    if sequence1 is None:
        sequence1 = keyPoints[t:t + slidingWindowsDimension]
    sequence1 = serializeKeyPointsSequence(sequence1)

    x = np.array([])
    y = np.array([])
    # slide the sliding window and perform the dtw distance
    for t in range(t, keyPoints.shape[0] - slidingWindowsDimension):
        sequence2 = keyPoints[t:t + slidingWindowsDimension]
        sequence2 = serializeKeyPointsSequence(sequence2)
        distance, path = dtw.fastdtw(sequence1, sequence2, dist=euclidean)
        x = np.append(x, t)
        y = np.append(y, distance)

    return findmin(10, x, y, 'Dtw', plotChart, slidingWindowsDimension, user)


def extractCyclesByEuclidean(slidingWindowsDimension, keyPoints, plotChart=False, sequence1=None):
    """
    Return the values in terms of frames that corresponds to the start and the end of cycles using the euclidean distance.
    The idea is to take a sliding window of a certain dimension (slidingWindowsDimension) that slides along the frames of
    the video and check the distance with the first window (the initial part of the video with dimensions slidingWindowDimesnion),
    then perform the euclidean and get, for each value of t (the shift value), the distance
    :param slidingWindowsDimension: int
    :param keyPoints: array of keypoints
    :param plotChart: boolean to decide if plot the chart of the cycle extraction
    :param sequence1: array. Used to extract cycle of the user
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

    return findmin(10, x, y, 'Euclidean', plotChart, slidingWindowsDimension, user=False)


def getMeanMeasures(keyPointsSequence, meanRange):
    """
    Return the mean measures in a neighborhood(meanRange) of the central frame of the torso distance (for scale normalization),
    midhip x and midhip y (for traslation normalization)
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


# pass keypoints to test it!
def getPointsAngles(means, weights):
    """
    For each frame retrieve the angle between the joints specified in the 'jointsNumber' list.
    To get cosine of the angle the dot product formula is used, then the arccos returns the angle
    :param means: array of the mean poses
    :param weights: weights of the joints
    :return: array of the joints angles
    """
    jointsNumber = {'leftUpperBody': [[8, 1], [1, 2], [2, 3], [3, 4]],
                    'rightUpperBody': [[8, 1], [1, 5], [5, 6], [6, 7]],
                    'leftLowerBody': [[1, 8], [8, 9], [9, 10], [10, 11], [11, 23]],
                    'rightLowerBody': [[1, 8], [8, 12], [12, 13], [13, 14], [14, 20]], }

    angles = []
    for pose in means:
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
                # vector normalizationq
                bodyVector1 = np.divide(bodyVector1, np.linalg.norm(bodyVector1))
                bodyVector2 = np.divide(bodyVector2, np.linalg.norm(bodyVector2))
                # "inverse" dot product
                angle = np.degrees(np.arccos(np.clip((np.dot(bodyVector1, bodyVector2)), -1.0, 1.0)))
                # string to identify the calculated angle
                strAngle = str(jointIndex[1]) + "-" + str(jointIndex[0]) + "|" + \
                           str(jointIndexN[0]) + "-" + str(jointIndexN[1])
                # list of: joint linked to neck, angle formed with torso, each long 25
                frameAngles.append([strAngle, angle])
        angles.append(frameAngles)
    return np.asarray(angles)


def checkByAngles(trainerMeans, userMeans, weights, errorRange, path):
    """
    For each keypoint the angle of the trainer is compared with the angle of the user
    :param trainerMeans: array. Mean poses of the trainer
    :param userMeans: array. Mean poses of the user
    :param weights: array. Weights of the joint
    :param errorRange: error allowed
    :param path: path of the aligned user trainer
    :return:
    """
    angles = getPointsAngles(trainerMeans, weights)
    userAngles = getPointsAngles(userMeans, weights)
    wrongPoses = []
    wrongPosesIndex = [0] * len(path)
    for couple in range(len(path)):
        trainerPose = path[couple][0]
        userPose = path[couple][1]
        error = 0
        keypointsWrong = []
        # take a random pose to get the number of keypoints involved. This number can change due to the weights
        for keypoint in range(min(len(userAngles[userPose]), len(angles[trainerPose]))):
            # note that at this point the keypoint 0 is the first angle calculated
            trainerAngle = angles[trainerPose][keypoint][1]
            userAngle = userAngles[userPose][keypoint][1]
            if userAngle > trainerAngle + errorRange or userAngle < trainerAngle - errorRange:
                error += weights[int(angles[trainerPose][keypoint][0][0])][1]
                keypointsWrong.append(userAngles[userPose][keypoint][0])
        if error > 2:
            wrongPoses.append([trainerPose, keypointsWrong, 'Error weight: ' + str(round(error, 3))])
            wrongPosesIndex[couple] = - 1
    # print('trainer angles:', angles)
    # print('user angles:', userAngles)
    return wrongPoses, wrongPosesIndex


def getGramDistance(pose_descriptor_1, pose_descriptor_2):
    """

    :param pose_descriptor_1:
    :param pose_descriptor_2:
    :return:
    """
    g1 = np.outer(pose_descriptor_1, pose_descriptor_1.T)
    g2 = np.outer(pose_descriptor_2, pose_descriptor_2.T)
    # print('Printing matrix shapes'), print(pose_descriptor_1.shape), print(g1.shape)
    distance = np.trace(g1) + np.trace(g2) - \
               2.0 * np.trace(la.fractional_matrix_power(
        np.dot(la.fractional_matrix_power(g1, 0.5), np.dot(g2, la.fractional_matrix_power(g1, 0.5))), 0.5))
    return distance


def getDescriptor(Pose, means, angles):
    """

    :param Pose:
    :param means:
    :param angles:
    :return:
    """
    poseDescriptor = np.empty(shape=[25, 2])
    for i in range(0, len(means[Pose])):
        poseDescriptor[i][0] = means[Pose][i][0]
        if not angles:
            poseDescriptor[i][1] = means[Pose][i][1]
    return poseDescriptor


def checkByGramMatrix(path, trainerMeans, userMeans, distance_error, angles):
    """

    :param path:
    :param trainerMeans:
    :param userMeans:
    :param distance_error:
    :param angles:
    :return:
    """
    posesWrong = []
    for couple in range(len(path)):
        trainerPose = path[couple][0]
        userPose = path[couple][1]
        pose_descriptor_trainer = getDescriptor(trainerPose, trainerMeans, angles)
        pose_descriptor_user = getDescriptor(userPose, userMeans, angles)
        distance = getGramDistance(pose_descriptor_trainer, pose_descriptor_user)
        if distance > distance_error:
            posesWrong.append([trainerPose, userPose])  # number of errors is equal to the length of the poseWrong list
    return posesWrong
