import numpy as np
import scipy.io
import fastdtw
from DevelopmentScripts.Utility import printProgressBar
from DevelopmentScripts.OPW import opw

def optimization(X, z, classNum, templateNum, l):
    """
    :param X: np array containing in each element a training sequence
    :param z: np array containing in each element the class of the corresponding sequence in X
    :param classNum: the number of possible classes
    :param templateNum: DUNNO.
    :param l: lambda value
    :return: the value of W
    """
    # create virtual sequences
    classNum = len(X)
    err_limit = 0.000001
    trainsetnum = np.zeros(shape=(classNum))
    V = []
    activedim = 0
    downdim = classNum * templateNum
    dim = X[0][0].shape[1]
    N = sum(trainsetnum)
    for c in range(classNum):
        trainsetnum[c] = len(X[c])
        V.append(np.zeros(shape=(templateNum, downdim)))
        for a in range(templateNum):
            V[c][a][activedim] = 1
            activedim += 1
    # initialize the alignment matrices
    T = []
    for c in range(classNum):
        tmpT = []
        for n in range(len(X[c])):
            # tmpT = np.zeros(shape=(X[n].shape[0],V[n].shape[0]))
            # m = min(X[n].shape[0],V[n].shape[0])
            # for i in range(m):
            #     tmpT[i][i] = 1
            seqLen = X[c][n].shape[0]
            tmpT.append(np.ones(shape=(seqLen, templateNum)) / (seqLen * templateNum))
        T.append(tmpT)

    loss_old = 10 ^ 8
    maxIterations = 1000
    for k in range(maxIterations):
        printProgressBar(k, maxIterations, 'Iteration: ' + str(k) + '/' + str(maxIterations))
        loss = 0
        W_a = np.zeros(shape=(dim, dim))
        W_b = np.zeros(shape=(dim, downdim))
        for c in range(classNum):
            for n in range(len(X[c])):
                for i in range(X[c][n].shape[0] - 1):
                    for j in range(V[c].shape[0] - 1):
                        W_a += T[c][n][i][j] * np.expand_dims(X[c][n][i], 1).dot(
                            np.transpose(np.expand_dims(X[c][n][i], 1)))
                        v = np.transpose(np.expand_dims(V[c][j], 1))
                        W_b += T[c][n][i][j] * np.expand_dims(X[c][n][i], 1).dot(v)
            # update W
            W_a = W_a + l * N * np.identity(dim)
            # W = np.divide(W_a,W_b)
            W = np.linalg.solve(W_a, W_b)
            # print(W)
            for n in range(len(X[c])):
                # update T
                # d,path = myDTW.dtw(X[c][n].dot(W),V[c])
                d, path = fastdtw.fastdtw(X[c][n].dot(W), V[c])
                # d, path = opw(X[c][n].dot(W), V[c], None, None, lambda1=50, lambda2=12.1, sigma=1, VERBOSE=0)
                loss = loss + d
                T[c][n] = pathToMat(path, T[c][n])
            loss = loss / N + np.trace(W.dot(np.transpose(W)))
            if abs(loss - loss_old) < err_limit:
                break
            else:
                loss_old = loss
    return W


def pathToMat(path, T):
    mat = np.zeros(shape=(T.shape[0], T.shape[1]))
    for el in path:
        mat[el[0], el[1]] = 1
    return mat

# X0 = []
# X1 = []
# X = []
# Y = []
# keyPoints = np.load('./KeyPoints/Trainer/arm-clap.npy')
# sequence1 = keyPoints[0:50,:,0]
# sequence2 = keyPoints[75:130,:,0]
# sequence3 = keyPoints[150:205,:,0]
# keyPoints = np.load('./KeyPoints/Trainer/squat0.npy')
# sequence4 = keyPoints[0:100,:,0]
# sequence5 = keyPoints[110:200,:,0]
# sequence6 = keyPoints[220:310,:,0]
# X0.append(sequence1)
# X0.append(sequence2)
# X0.append(sequence3)
# X1.append(sequence4)
# X1.append(sequence5)
# X1.append(sequence6)
# X.append(X0)
# X.append(X1)
# Y.append(0)
# Y.append(0)
# Y.append(0)
# Y.append(1)
# Y.append(1)
# Y.append(1)
# # X = np.asarray(X)
# # z = np.asarray(Y)
#
# # scipy.io.savemat('train.mat', mdict={'arr': X})
# # scipy.io.savemat('labels.mat', mdict={'arr': z})
# W = optimization(X,Y,2,4,0.01)
# print(W)
