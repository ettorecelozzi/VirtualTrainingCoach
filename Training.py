import numpy as np
import os
from DevelopmentScripts.metricLearning import optimize
from DevelopmentScripts import PoseAnalysis
from DevelopmentScripts import Normalization as norm

pathToTest = './Dataset/test/'


def init_trainset(pathToTrain):
    """
    Build the correct normalized training set structure given the dataset
    :return: training set, shape=(#classes, #samples, #sequences, d)
    """
    # training set initialization
    exerciseFolders = os.listdir(pathToTrain)
    trainingSet = []
    for folder in exerciseFolders:
        exerciseExecutions = os.listdir(pathToTrain + folder)
        classExercise = []
        for exercise in exerciseExecutions:
            if 'confidence' in exercise: continue
            keypoints = np.load(pathToTrain + folder + '/' + exercise)
            # keypoints normalization
            meanTorso, meanHipX, meanHipY = PoseAnalysis.getMeanMeasures(keypoints, 50)
            keypoints = norm.normalize(meanTorso, meanHipX, meanHipY, keypoints.copy())

            classExercise.append(keypoints.reshape((keypoints.shape[0], keypoints.shape[1] * keypoints.shape[2])))

        trainingSet.append(classExercise)
    return trainingSet


def train(pathToSet, align_algorithm):
    """
    Compute the parameters matrix M and store the factorization W. M = W*W'
    :return: matrix M, shape=(d,d) with d dimension of pose
    """
    # training
    if 'W_' + align_algorithm + '.npy' not in os.listdir(pathToSet):
        trainingSet = init_trainset(pathToSet + '/Keypoints/')
        W = optimize(trainingSet, templateNum=4, l=0.01, err_limit=0.01, align_algorithm=align_algorithm)
        np.save(pathToSet + 'W_' + align_algorithm + '.npy', W)
    else:
        W = np.load(pathToSet + 'W_' + align_algorithm + '.npy')
    M = np.dot(W, np.transpose(W))
    return M
