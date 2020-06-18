import numpy as np
import fastdtw
from DevelopmentScripts.OPW import opw
from DevelopmentScripts.Utility import printProgressBar


def optimize(trainset, templateNum, l, err_limit, align_algorithm="dtw", lambda1=10, lambda2=0.5, sigma=12):
    """
    :param trainset: list containing the training sequences divided by class
                        trainset: a list with size of (1, the number of classes);
                        for the c-th class, trainset[c]: a list with size of (1, the number of training sequences for the c-th class)
                        for the i-th sequence sample, trainset[c][i]: a numpy matrix with size of (the length of the sequence, the dimensionality of vectors in this sequence)
    :param templateNum: dimensions of virtual sequences
    :param l: lambda value
    :param err_limit: bottom limit for the error to stop the optimization
    :param align_algorithm: algorithm used to align sequences "dtw" or "opw"
    :param lambda1: lambda 1 value used in OPW
    :param lambda2: lambda 2 value used in OPW
    :param sigma: sigma value used in OPW
    :return: the value of L in the decomposizìtion of M in mahalanobis distance
    """
    # create virtual sequences V
    classNum = len(trainset)
    trainsetnum = np.zeros(shape=(classNum))
    V = []
    activedim = 0
    downdim = classNum * templateNum
    dim = trainset[0][0].shape[1]
    N = sum(trainsetnum)
    for c in range(classNum):
        trainsetnum[c] = len(trainset[c])
        V.append(np.zeros(shape=(templateNum, downdim)))
        for a in range(templateNum):
            V[c][a][activedim] = 1
            activedim += 1
    # initialize the alignment matrices T
    T = []
    for c in range(classNum):
        tmpT = []
        for n in range(len(trainset[c])):
            seqLen = trainset[c][n].shape[0]
            tmpT.append(np.ones(shape=(seqLen, templateNum)) / (seqLen * templateNum))
        T.append(tmpT)

    # optimization
    loss_old = 10 ^ 8
    maxIterations = 1000
    for k in range(maxIterations):
        printProgressBar(k, maxIterations, 'Iteration: ' + str(k) + '/' + str(maxIterations))
        loss = 0
        L_a = np.zeros(shape=(dim, dim))
        L_b = np.zeros(shape=(dim, downdim))
        for c in range(classNum):
            for n in range(len(trainset[c])):
                for i in range(trainset[c][n].shape[0] - 1):
                    for j in range(V[c].shape[0] - 1):
                        L_a += T[c][n][i][j] * np.expand_dims(trainset[c][n][i], 1).dot(np.transpose(np.expand_dims(trainset[c][n][i], 1)))
                        v = np.transpose(np.expand_dims(V[c][j], 1))
                        L_b += T[c][n][i][j] * np.expand_dims(trainset[c][n][i], 1).dot(v)
            # update L
            L_a = L_a + l * N * np.identity(dim)
            L = np.linalg.solve(L_a, L_b)  # instead inverting L_a solve the linear system
            for n in range(len(trainset[c])):
                # update T
                if align_algorithm == "dtw":
                    d, path = fastdtw.fastdtw(trainset[c][n].dot(L), V[c])
                    T[c][n] = pathToMat(path, T[c][n])
                else:
                    d, T[c][n] = opw(trainset[c][n].dot(L), V[c], a=None, b=None, lambda1=lambda1, lambda2=lambda2,
                                   sigma=sigma, VERBOSE=0)
                loss = loss + d
            loss = loss / N + np.trace(L.dot(np.transpose(L)))
            if abs(loss - loss_old) < err_limit:
                break
            else:
                loss_old = loss
    return L


def pathToMat(path, T):
    mat = np.zeros(shape=(T.shape[0], T.shape[1]))
    for el in path:
        mat[el[0], el[1]] = 1
    return mat
