import numpy as np
import os
from DevelopmentScripts.metricLearning import optimize
from collections import defaultdict
from DevelopmentScripts import PoseAnalysis
from DevelopmentScripts import Normalization as norm
from fastdtw import fastdtw
from Training import train
import random
from DevelopmentScripts.OPW import opw, transport_vector_to_path
import pandas as pd
import statistics
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import DistanceMetric
import csv

pathToTrain = './Dataset/train/'
pathToTest = './Dataset/test/'
dataset = './MSRDataset/'


def mahalanobis_like_distance(X, Y, M):
    """
    Mahlanobis like distance between two sequences given the parameters matrix M
    :param X: sequence, shape=(#joints * #coordinates)
    :param Y: sequence, shape=(#joints * #coordinates)
    :param M: matrix, shape=(#joints * #coordinates, #joints * #coordinates)
    :return: distance between each sequence, float
    """
    # sqrt slow down, maybe can be removed
    dist = (np.sqrt(np.dot(np.transpose(np.dot(M, X) - np.dot(M, Y)), np.dot(M, X) - np.dot(M, Y))))
    return dist


def get_dist_T(align_algorithm, X, Y, M):
    """
    Compute the distance and path given the strategy and the sequences
    :param align_algorithm: dtw or opw
    :param X: sequence 1
    :param Y: sequence 2
    :param M: learning matrix
    :return: dist, path
    """
    if align_algorithm == 'dtw':
        dist, path = fastdtw(np.dot(X, M), np.dot(Y, M))
    elif align_algorithm == 'opw':
        dist, T = opw(np.dot(X, M), np.dot(Y, M), a=None, b=None, lambda1=50, lambda2=2.0, sigma=1, VERBOSE=0)
        path = transport_vector_to_path(T)
    else:
        raise Exception('Align strategy not recognized')
    return dist, path


def distance_between_sequences(X, Y, M, align_algorithm):
    # Keypoints normalization
    meanTorso, meanHipX, meanHipY = PoseAnalysis.getMeanMeasures(X, 50)
    X = norm.normalize(meanTorso, meanHipX, meanHipY, X.copy())
    X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))

    # Keypoints normalization
    meanTorso, meanHipX, meanHipY = PoseAnalysis.getMeanMeasures(Y, 50)
    Y = norm.normalize(meanTorso, meanHipX, meanHipY, Y.copy())
    Y = Y.reshape((Y.shape[0], Y.shape[1] * Y.shape[2]))

    resultDistances = []
    resultDistancesE = []
    dist, path = get_dist_T(align_algorithm, X, Y, M)
    return dist
    # for el in path:
    #     # MISSING metric that compute distance between joints
    #     resultDistances.append(mahalanobis_like_distance(X[el[0]], Y[el[1]], M))
    #     resultDistancesE.append(np.linalg.norm(X[el[0]] - Y[el[1]]))
    # return resultDistances


def test_in_same_class(M, align_algorithm):
    """
    Compute the distance between a test file and a train file of the trainerSequences
    :param M: parameters matrix from the train, shape=(#joints * #coordinates, #joints * #coordinates)
    :param align_algorithm: use opw or dtw
    :return: dictionary of the distances between each test sequence and the reference train sequence
    """
    # dictionary that contain the train sequences to compare to the test sequences
    trainerSequences = {'arm-clap': 'arm-clap_c0.npy', 'double-lunges': 'double-lunges_c0.npy',
                        'dumbbell-curl': 'dumbbell-curl_c0.npy', 'push-ups0': 'push-ups0_c0.npy',
                        'push-ups45': 'push-ups45_c0.npy', 'push-ups90': 'push-ups90_c0.npy',
                        'single-lunges': 'single-lunges_c0.npy', 'squat0': 'squat0_c0.npy',
                        'squat45': 'squat45_c0.npy', 'squat90': 'squat90_c0.npy'}
    exerciseFolders = os.listdir(pathToTest)
    resultDistances = defaultdict(lambda: defaultdict(list))
    for folder in exerciseFolders:
        exerciseExecutions = os.listdir(pathToTest + folder)
        for exercise in exerciseExecutions:
            X = np.load(pathToTrain + folder + '/' + trainerSequences[folder])
            # Keypoints normalization
            meanTorso, meanHipX, meanHipY = PoseAnalysis.getMeanMeasures(X, 50)
            X = norm.normalize(meanTorso, meanHipX, meanHipY, X.copy())
            X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))

            Y = np.load(pathToTest + folder + '/' + exercise)
            # Keypoints normalization
            meanTorso, meanHipX, meanHipY = PoseAnalysis.getMeanMeasures(Y, 50)
            Y = norm.normalize(meanTorso, meanHipX, meanHipY, Y.copy())
            Y = Y.reshape((Y.shape[0], Y.shape[1] * Y.shape[2]))

            dist, path = get_dist_T(align_algorithm, X, Y, M)
            resultDistances[folder][exercise] = dist
    return resultDistances


def test_different_class(M, numberOfTests, align_algorithm, pathToSet=pathToTest):
    """
    Given two sequences X,Y that belong to two different exercise class compute the difference
    :param M: parameters matrix from the train, shape=(#joints * #coordinates, #joints * #coordinates)
    :param align_algorithm: use opw or dtw
    :param numberOfTests: number of tests to perform, int
    :param pathToSet: path of the test or train folder
    """
    folders = os.listdir(pathToSet)
    for i in range(numberOfTests):
        randomFolder1 = random.choice(folders)
        randomFolder2 = random.choice(folders)
        while randomFolder1 == randomFolder2: randomFolder2 = random.choice(folders)
        randomExercise1 = random.choice(os.listdir(pathToSet + randomFolder1 + '/'))
        randomExercise2 = random.choice(os.listdir(pathToSet + randomFolder2 + '/'))
        X = np.load(pathToSet + randomFolder1 + '/' + randomExercise1)
        Y = np.load(pathToSet + randomFolder2 + '/' + randomExercise2)

        # X = np.load(pathToTrain + 'arm-clap/arm-clap_1_c0.npy')
        # Y = np.load(pathToTrain + 'arm-clap/arm-clap_1_c1.npy')

        # Keypoints normalization
        meanTorso, meanHipX, meanHipY = PoseAnalysis.getMeanMeasures(X, 50)
        X = norm.normalize(meanTorso, meanHipX, meanHipY, X.copy())
        X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))

        # Keypoints normalization
        meanTorso, meanHipX, meanHipY = PoseAnalysis.getMeanMeasures(Y, 50)
        Y = norm.normalize(meanTorso, meanHipX, meanHipY, Y.copy())
        Y = Y.reshape((Y.shape[0], Y.shape[1] * Y.shape[2]))

        dist, path = get_dist_T(align_algorithm, X, Y, M)
        print(f'Distance between **{randomExercise1.split(".")[0]}** and **{randomExercise2.split(".")[0]}**: {dist}')


def confusion_matrix(M, pathToSet, align_algorithm):
    """
    Compare one sample of each class with one sample of each class
    :param M: parameters matrix from the train, shape=(#joints * #coordinates, #joints * #coordinates)
    :param align_algorithm: use opw or dtw
    :param pathToSet: path of the test or train folder
    :return: confusion matrix, shape=(#classes, #classes)
    """
    folders = os.listdir(pathToSet)
    matrix = defaultdict(lambda: defaultdict(int))
    for folder in folders:
        exercise = os.listdir(pathToSet + folder + '/')[0]  # take the first sample for each class
        X = np.load(pathToSet + folder + '/' + exercise)
        meanTorso, meanHipX, meanHipY = PoseAnalysis.getMeanMeasures(X, 50)
        X = norm.normalize(meanTorso, meanHipX, meanHipY, X.copy())
        X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))

        for folder1 in folders:
            exercise1 = os.listdir(pathToSet + folder1 + '/')[0]  # take the first sample for each class
            Y = np.load(pathToSet + folder1 + '/' + exercise1)
            meanTorso, meanHipX, meanHipY = PoseAnalysis.getMeanMeasures(Y, 50)
            Y = norm.normalize(meanTorso, meanHipX, meanHipY, Y.copy())
            Y = Y.reshape((Y.shape[0], Y.shape[1] * Y.shape[2]))

            dist, path = get_dist_T(align_algorithm, X, Y, M)
            matrix[folder][folder1] = dist
    return matrix


# training
# align_algorithm = 'opw'
# M = train(pathToSet=dataset, align_algorithm=align_algorithm)
# for align_algorithm in ['dtw','opw']:
#     M = train(pathToSet=dataset, align_algorithm=align_algorithm)

# conf_matrix = confusion_matrix(M, dataset + 'Keypoints/', align_algorithm)
# dataframe = pd.DataFrame.from_dict(conf_matrix, orient='index')
# pd.set_option('display.max_columns', None)
# print(dataframe)

# # test in the same class
# resultDistances = test_in_same_class(M, align_algorithm)
# for key in resultDistances.keys():
#     print(resultDistances[key])
#
# # test between different classes
# test_different_class(M, numberOfTests=6, align_algorithm=align_algorithm, pathToSet=pathToTest)
#
# # distance between each joints
# X = np.load(pathToTrain + 'arm-clap/arm-clap_1_c0.npy')
# Y = np.load(pathToTrain + 'arm-clap/arm-clap_1_c1.npy')
# distance_between_sequences(X, Y, M, align_algorithm)
#
# exercises = ['arm-clap','double-lunges','single-lunges','dumbbell-curl','push-ups0','push-ups45','push-ups90', 'squat0','squat45','squat90']
# for exercise in exercises:
#     for align_algorithm in ['dtw','opw']:
#         print(exercise + align_algorithm + '\n')
#         train(pathToSet=pathToTrain,align_algorithm=align_algorithm,exercise=exercise)
# exercises = ['arm-clap','double-lunges','single-lunges','dumbbell-curl','push-ups0','push-ups45','push-ups90', 'squat0','squat45','squat90']
# for exercise in exercises:
#     print("----------" + exercise + '----------')
#     X = np.load(pathToTrain + exercise + '/good/' + exercise + '_0_c1.npy')
#     for align_algorithm in ['dtw','opw']:
#         print(align_algorithm + '---')
#         W = np.load('./Dataset/OnevsAll/NoBaseline/' + 'W_' + exercise + '_' + align_algorithm + '.npy')
#         M = np.dot(W, np.transpose(W))
#         testSamples = os.listdir(pathToTest + exercise + '/')
#         for type in testSamples:
#             distList = []
#             samples = os.listdir(pathToTest + exercise + '/' + type + '/')
#             for sample in samples:
#                 Y = np.load(pathToTest + exercise + '/' + type + '/' + sample)
#                 d = distance_between_sequences(X,Y,M,align_algorithm)
#                 # print(type + ": " + str(d))
#                 if align_algorithm == "dtw":
#                     distList.append(d)
#                 else:
#                     distList.append(d[0])
#             mean = statistics.mean(distList)
#             std = statistics.stdev(distList)
#             print("mean " + type + ": " + str(mean) + "    -    std: " + str(std))
#     print("\n")


def knn(k):
    exercises = ['arm-clap', 'double-lunges', 'single-lunges', 'dumbbell-curl', 'push-ups0', 'push-ups45', 'push-ups90',
                 'squat0', 'squat45', 'squat90']
    for exercise in exercises:  # per ogni esercizio

        print("----------" + exercise + '----------')
        for align_algorithm in ['dtw', 'opw']:  # per ogni tipo di algoritmo di allineamento
            print(align_algorithm + '---')
            correctlyclassified = 0
            wronglyclassified = 0

            listcorrectlyclassified = []  # lists for each different k
            listwronglyclassified = []
            listconfmatrix = []
            for i in range(1, k, 2):
                listcorrectlyclassified.append(0)
                listwronglyclassified.append(0)
                confm = np.zeros(shape=(2, 2))
                listconfmatrix.append(confm)

            W = np.load('./Dataset/OnevsAll/Baseline/' + 'W_' + exercise + '_' + align_algorithm + '.npy')
            M = np.dot(W, np.transpose(W))
            testSamples = os.listdir(pathToTest + exercise + '/')
            for type in testSamples:
                samples = os.listdir(pathToTest + exercise + '/' + type + '/')

                for sample in samples:  # per ogni esempio di test
                    knearest = np.full(k, np.inf)
                    knearestclass = []
                    for i in range(k):
                        knearestclass.append('good')
                    Y = np.load(pathToTest + exercise + '/' + type + '/' + sample)
                    ###find k nearest neighbors
                    trainSamples = os.listdir(pathToTrain + exercise + '/')
                    for trainType in trainSamples:
                        tsamples = os.listdir(pathToTrain + exercise + '/' + trainType + '/')
                        for tsample in tsamples:
                            X = np.load(pathToTrain + exercise + '/' + trainType + '/' + tsample)
                            if align_algorithm == "dtw":
                                d = distance_between_sequences(X, Y, M, align_algorithm)
                            else:
                                d = distance_between_sequences(X, Y, M, align_algorithm)[0]
                            m = np.max(knearest)
                            if np.isinf(m) or d < m:
                                idx = np.where(knearest == m)[0][0]
                                knearest[idx] = d
                                knearestclass[idx] = trainType
                    for kk in range(1, k, 2):
                        countgood = 0
                        countwrong = 0
                        idxs = knearest.argsort()[:kk]
                        T = [knearestclass[i] for i in idxs]
                        for el in T:
                            if el == "good":
                                countgood += 1
                            else:
                                countwrong += 1
                        if countgood > countwrong:
                            if type == "good":
                                listcorrectlyclassified[kk // 2] += 1
                                listconfmatrix[kk // 2][0][0] += 1
                            else:
                                listwronglyclassified[kk // 2] += 1
                                listconfmatrix[kk // 2][0][1] += 1
                        else:
                            if type == "wrong":
                                listcorrectlyclassified[kk // 2] += 1
                                listconfmatrix[kk // 2][1][1] += 1
                            else:
                                listwronglyclassified[kk // 2] += 1
                                listconfmatrix[kk // 2][1][0] += 1

            for kk in range(1, k, 2):
                accuracy = listcorrectlyclassified[kk // 2] / (
                        listcorrectlyclassified[kk // 2] + listwronglyclassified[kk // 2])
                # accuracy = correctlyclassified / (correctlyclassified + wronglyclassified)
                print("accuracy k= " + str(kk) + ': ' + str(accuracy))
                print(listconfmatrix[kk // 2])
                print("\n")


knn(15)

def distance_between_sequences_KNN(X, Y, **kwargs):
    # Keypoints normalization
    meanTorso, meanHipX, meanHipY = PoseAnalysis.getMeanMeasures(X, 50)
    X = norm.normalize(meanTorso, meanHipX, meanHipY, X.copy())
    X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))

    # Keypoints normalization
    meanTorso, meanHipX, meanHipY = PoseAnalysis.getMeanMeasures(Y, 50)
    Y = norm.normalize(meanTorso, meanHipX, meanHipY, Y.copy())
    Y = Y.reshape((Y.shape[0], Y.shape[1] * Y.shape[2]))

    align_algorithm = kwargs['metric_params']['align_algorithm']
    M = kwargs['metric_params']['M']
    dist, path = get_dist_T(align_algorithm, X=X, Y=Y, M=M)
    return dist


def KNNScipy(exercise, k, align_algorithm):
    namesTrain, Ytrain = [], []
    with open("./DatasetML/train/" + exercise + ".csv") as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        for row in reader:
            namesTrain.append(row[0])  # name of the keypoints files
            Ytrain.append(row[1])  # label of the file, 0 wrong 1 good
    Xtrain = [np.load("./DatasetML/train/" + exercise.capitalize() + "/" + name + ".npy") for name in namesTrain]
    print(len(Xtrain), len(Xtrain[0]), len(Xtrain[0][0]), len(Xtrain[0][0][0]))

    namesTest, Ytest = [], []
    with open("./DatasetML/test/" + exercise + ".csv") as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        for row in reader:
            namesTest.append(row[0])  # name of the keypoints files
            Ytest.append(row[1])  # label of the file, 0 wrong 1 good
    Xtest = [np.load("./DatasetML/test/" + exercise.capitalize() + "/" + name + ".npy") for name in namesTest]
    print(len(Xtest), len(Ytest))

    W = np.load("./DatasetML/train/W_" + exercise + "_" + align_algorithm + ".npy")
    M = np.dot(W, np.transpose(W))
    knn = KNeighborsClassifier(n_neighbors=k, metric=distance_between_sequences_KNN,
                               metric_params={"align_algorithm": align_algorithm, "M": M})
    knn.fit(Xtrain, Ytrain)


# KNNScipy("arm-clap", 5, "dtw")
