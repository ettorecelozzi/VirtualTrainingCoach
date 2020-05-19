import numpy as np
import os
from DevelopmentScripts.metricLearning import optimization
from collections import defaultdict
import pandas as pd
from DevelopmentScripts import PoseAnalysis
from DevelopmentScripts import Normalization as norm


def mahalanobis_like_distance(X, Y, L):
    """
    Mahlanobis like distance between two array of sequences given the parameters matrix M
    :param X: sequences, shape=(N,d)
    :param Y: sequences, shape=(N,d)
    :param M: matrix, shape=(d,d)
    :return: distance between each sequence, shape=(N)
    """
    dist = []
    for x, y in zip(X, Y):
        dist.append(np.dot(np.dot(np.transpose(np.subtract(x, y)), L), np.subtract(x, y)))

    # sqrt slow down a lot, maybe can be removed
    # dist = (np.sqrt(np.dot(np.transpose(np.dot(L, X) - np.dot(L, Y)),
    #                        np.dot(L, X) - np.dot(L, Y))))
    return dist


# training set initialization
pathToTrain = './Dataset/train/'
exerciseFolders = os.listdir(pathToTrain)
trainingSet = []
for folder in exerciseFolders:
    exerciseExecutions = os.listdir(pathToTrain + folder)
    classExercise = []
    for exercise in exerciseExecutions:
        keypoints = np.load(pathToTrain + folder + '/' + exercise)
        # keypoints normalization
        meanTorso, meanHipX, meanHipY = PoseAnalysis.getMeanMeasures(keypoints, 50)
        keypoints = norm.normalize(meanTorso, meanHipX, meanHipY, keypoints.copy())

        classExercise.append(keypoints.reshape((keypoints.shape[0], keypoints.shape[1] * keypoints.shape[2])))

    trainingSet.append(classExercise)

# training
# W = optimization(trainingSet, 0, 0, 4, 0.01)
# np.save('./Dataset/W_opw_kps_normalized.npy', W)
W = np.load('./Dataset/W_opw_kps_normalized.npy')
L = np.dot(W, np.transpose(W))

# test
trainerSequences = {'arm-clap': 'arm-clap_c0.npy', 'double-lunges': 'double-lunges_c0.npy',
                    'dumbbell-curl': 'dumbbell-curl_c0.npy', 'push-ups0': 'push-ups0_c0.npy',
                    'push-ups45': 'push-ups45_c0.npy', 'push-ups90': 'push-ups90_c0.npy',
                    'single-lunges': 'single-lunges_c0.npy', 'squat0': 'squat0_c0.npy',
                    'squat45': 'squat45_c0.npy', 'squat90': 'squat90_c0.npy'}

pathToTest = './Dataset/test/'
resultDistances = defaultdict(lambda: defaultdict(list))
for folder in exerciseFolders:
    exerciseExecutions = os.listdir(pathToTest + folder)
    for exercise in exerciseExecutions:
        Y = np.load(pathToTest + folder + '/' + exercise)
        # Keypoints normalization
        meanTorso, meanHipX, meanHipY = PoseAnalysis.getMeanMeasures(Y, 50)
        Y = norm.normalize(meanTorso, meanHipX, meanHipY, Y.copy())
        Y = Y.reshape((Y.shape[0], Y.shape[1] * Y.shape[2]))

        X = np.load('./Dataset/train/' + folder + '/' + trainerSequences[folder])
        # Keypoints normalization
        meanTorso, meanHipX, meanHipY = PoseAnalysis.getMeanMeasures(X, 50)
        X = norm.normalize(meanTorso, meanHipX, meanHipY, X.copy())
        X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))

        resultDistances[folder][exercise] = mahalanobis_like_distance(X, Y, L)

# dataframe = pd.DataFrame.from_dict(resultDistances, orient='index')
# print(dataframe)
for key in resultDistances.keys():
    print(resultDistances[key])
