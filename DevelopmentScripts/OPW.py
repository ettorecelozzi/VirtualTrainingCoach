import numpy as np
from scipy.spatial.distance import cdist


def opw(x, y, a=None, b=None, lambda1=10, lambda2=0.5, sigma=12, M=None, VERBOSE=1):
    """
    OPW aligning
    :param x: first sequence to align. shape=(N1,d) d > 2
    :param y: second sequence to align. shape=(N2,d) d > 2
    :param a: weights of the first points sequence. If none uniform weights are used
    :param b: weights of the second points sequence. If none uniform weights are used
    :param lambda1: regularization term for the inverse difference moment (to be tuned)
    :param lambda2: regularization term for Kullback-Leibler (KL) (to be tuned)
    :param sigma: variance of the gaussian distribution
    :param M: learning matrix used to compute distance between sequences
    :param VERBOSE: whether display the iteration status
    :return: distance (float value), transport matrix T, shape=(N1,N2)
    """
    tol = 0.005
    max_iter = 20  # higher means more accurate transport vector
    N, M = x.shape[0], y.shape[0]
    if (x.size / x.shape[0]) != (y.size / y.shape[0]):
        raise Exception(f'The dimension of the sequence must be the same, x: {x.shape} y: {y.shape}')

    p = np.zeros((N, M))
    s = np.zeros((N, M))

    # param in the middle on which center the gaussian (see later)
    mid_para = np.sqrt(1 / np.power(N, 2) + 1 / np.power(M, 2))
    for i in range(N):
        for j in range(M):
            diag = np.abs(i / N - j / M) / mid_para

            # Gaussian distribution centered at the intersection on the diagonal
            # (prior distribution of the transport matrix)
            p[i, j] = np.exp(-np.power(diag, 2) / 2 * np.power(sigma, 2)) / (sigma * np.sqrt(2 * np.pi))

            # constant defined at page 6 of the OPW paper
            s[i, j] = lambda1 / (np.power((i / N - j / M), 2) + 1)

    if M is None:
        # pairwise distance between x and y
        d = cdist(x, y, 'sqeuclidean')
    else:
        # Malhanobis like distance using the learning matrix
        d = cdist(x, y, lambda X, Y:
        (np.sqrt(np.dot(np.transpose(np.dot(M, X) - np.dot(M, Y)), (np.dot(M, X) - np.dot(M, Y))))))
    ''' 
    In cases the instances in sequences are not normalized and/or are very high-dimensional, the matrix D can be
    normalized or scaled as follows: D = D/max(max(D));  D = D/(10^2)
    '''

    # This formula has been taken from page 6 of OPW paper.
    k = p * np.exp((s - d) / lambda2)  # every operator "*", "/" means element wise
    '''
    With some parameters, some entries of K may exceed the matching-precision limit; in such cases, you may need
    to adjust the parameters, and/or normalize the input features in sequences or the matrix D; Please see the paper for
    details. In practical situations it might be a good idea to do the following: K(K<1e-100)=1e-100;
    '''

    if a is None: a = np.ones((N, 1)) / N
    if b is None: b = np.ones((M, 1)) / M

    ainvK = k / a
    compt = 0

    '''
    Dual-Sinkhorn divergence is the dual problem of the Sinkhorn Distance; this is cheaper to compute. In the new defined 
    problem (see page 4 of https://papers.nips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport.pdf)
    the value of P^lambda (the solution), thanks to the entropy regularization, can be computed as diag(u)*K*diag(v), 
    where u and v are two non-negative vectors of R^d uniquely defined up to a multiplicative factor.
    '''
    u = np.ones((N, 1)) / N  # initialization of left scaling factors
    v = None

    # Sinkhorn Distances
    while compt < max_iter:
        # for real value there aren't differences in the transpose operation ( a' vs a.' in Matlab)
        u = 1 / (np.dot(ainvK, (b / (np.dot(np.transpose(k), u)))))
        compt += 1

        # check the stopping criterion every 20 fixed point iterations
        if (compt % 20) == 1 or compt == max_iter:
            # split computations to recover right and left scalings.
            v = b / np.dot(np.transpose(k), u)  # main iteration of Sinkhorn's algorithm
            u = 1 / np.dot(ainvK, v)

            # check if the value of diag(u) * K *diag(v) is smaller than the tolerance value. Remember that this is the
            # solution of the Dual-Sinkhorn divergence problem
            criterion = np.linalg.norm(np.sum(np.abs(v * (np.dot(np.transpose(k), u)) - b), axis=0), np.inf)
            if criterion < tol:
                # print(f'Convergence reached, criterion: {criterion} tol: {tol}')
                break
            if np.isnan(criterion):
                raise Exception('NaN values have appeared during the fixed point iteration. This problem appears '
                                'because of insufficient machine precision when processing computations with a '
                                'regularization value of lambda2 that is too high. '
                                'Try again with a reduced regularization parameter lambda (1 or 2) or with a '
                                'thresholded metric matrix d.')

            compt += 1
            if VERBOSE > 0:
                print(f'Iteration: {compt}, Criterion: {criterion}')

    U = k * d
    dist = np.sum(u * np.dot(U, v), axis=0)
    T = np.transpose(v) * (u * k)

    return dist, T


def transport_vector_to_path(T):
    """
    Given a transport vector T generated aligning two sequences of dimension (N1,d), (N2,d), extract the sequences
    aligning path
    :param T: vector, shape=(N1,N2)
    :return: path, shape=(max(N1,N2),)
    """
    path = []
    if T.shape[0] > T.shape[1]:
        for s1 in range(T.shape[0]):
            path.append((s1, np.argmax(T[s1])))
    else:
        T = np.transpose(T)
        for s1 in range(T.shape[0]):
            path.append((np.argmax(T[s1]), s1))
    return path
