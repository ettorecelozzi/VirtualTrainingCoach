import numpy as np
import os
from DevelopmentScripts.metricLearning import optimization
from collections import defaultdict
import pandas as pd


def mahalanobis_like_distance(X, Y, L):
    """
    Mahlanobis like distance between two array of sequences given the parameters matrix M
    :param X: sequences, shape=(N,d)
    :param Y: sequences, shape=(N,d)
    :param M: matrix, shape=(d,d)
    :return: distance between each sequence, shape=()
    """
    dist = []
    for x, y in zip(X, Y):
        # sqrt slow down a lot, maybe can be removed
        # dist.append(np.sqrt(np.dot(np.transpose(np.dot(L, x) - np.dot(L, y)),
        #                            np.dot(L, x) - np.dot(L, y))))
        dist.append(np.dot(np.dot(np.transpose(np.subtract(x, y)), L), np.subtract(x, y)))
    return dist


# training set initialization
pathToTrain = './Dataset/train/'
exerciseFolders = os.listdir(pathToTrain)
# trainingSet = []
# for folder in exerciseFolders:
#     exerciseExecutions = os.listdir(pathToTrain + folder)
#     classExercise = []
#     for exercise in exerciseExecutions:
#         keypoints = np.load(pathToTrain + folder + '/' + exercise)
#         classExercise.append(keypoints.reshape((keypoints.shape[0], keypoints.shape[1] * keypoints.shape[2])))
#     trainingSet.append(classExercise)
#
# # training
# W = optimization(trainingSet, 0, 0, 4, 0.01)
# np.save('./Dataset/W.npy', W)
W = np.load('./Dataset/W.npy')
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
        keypoints = np.load(pathToTest + folder + '/' + exercise)
        Y = keypoints.reshape((keypoints.shape[0], keypoints.shape[1] * keypoints.shape[2]))
        X = np.load('./Dataset/train/' + folder + '/' + trainerSequences[folder])
        X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
        resultDistances[folder][exercise] = mahalanobis_like_distance(X, Y, L)

# dataframe = pd.DataFrame.from_dict(resultDistances, orient='index')
# print(dataframe)
for key in resultDistances.keys():
    print(resultDistances[key])
