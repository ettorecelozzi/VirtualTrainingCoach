import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import os
import numpy as np
from DevelopmentScripts.KeyPointsFromImage import plotKeyPointsImage
import cv2
import shutil


def serializeKeyPointsSequence(keyPointsSequence, weights=None):
    """
    The DTW is between two time sequences of points, but we have sequences of frames ( that are lists of points ) and we
    can't do the DTW between sequnces of sequences of points.
    Solotion: We serialize the sequence of frame, meaning that for each frame we decouple each (x,y) so that a frame that
    at the beginning is [(x1,y1)(x2,y2)...] will became [x1,y1,x2,y2,...] in that way the dtw is possible
    :param keyPointsSequence: matrix of keypoints. shape=(#frame,25,2)
    :param weights: weights list associated to body joints
    :return: keypoints sequences serialized
    """

    # return np.reshape(keyPointsSequence, (keyPointsSequence.shape[0],
    #                                       keyPointsSequence.shape[1] * keyPointsSequence.shape[2]))

    res = []
    for keyPointsFrame in keyPointsSequence:
        framePoints = []
        pointIndex = 0
        for point in keyPointsFrame:
            if weights is not None and weights[pointIndex][1] != 0.0:
                framePoints.append(point[0])
                framePoints.append(point[1])
            else:
                framePoints.append(point[0])
                framePoints.append(point[1])

            pointIndex += 1
        res.append(framePoints)
    res = np.array(res)
    return res


def plotKeyPointsPose(keypoints, userMeanKeypoints, path, videoname, min, userMin, op, opWrapper,
                      wrongPosesMeanSTDIndex,
                      wrongPosesAngIndex,
                      plotKeypointsLabel=False, plotHeadKeypoint=True, plotFeetKeypoint=True):
    """
    Function that given the aligned trainer-User path plot the poses of each one
    """
    plt.style.use('dark_background')
    # coordinates of the points to tie
    lines = [(0, 15), (0, 16), (0, 17), (0, 18), (0, 1), (1, 8), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (8, 9),
             (9, 10), (10, 11), (11, 24), (11, 22), (23, 22), (22, 24), (8, 12), (12, 13), (13, 14), (14, 19), (19, 20),
             (21, 19), (21, 14)]
    # index of the label to space
    separate = [11, 24, 23, 22, 14, 21, 19, 20]
    head = [15, 16, 17, 18]
    feet = [24, 23, 22, 21, 19, 20]
    countPose = 0
    for couple in range(len(path)):
        trainerPose = path[couple][0]
        userPose = path[couple][1]
        trainerKeypoints = keypoints[trainerPose]  # frame keypoints for the trainerPose
        userKeypoints = userMeanKeypoints[userPose]
        pointToMerge = []
        pointToMergeUser = []
        for i in range(len(lines)):
            if plotHeadKeypoint is False and (lines[i][0] in head or lines[i][1] in head): continue
            if plotFeetKeypoint is False and (lines[i][0] in feet or lines[i][1] in feet): continue
            segment = [(trainerKeypoints[lines[i][0]][0], trainerKeypoints[lines[i][0]][1]),  # (x1,y1) (x2,y2) segment
                       (trainerKeypoints[lines[i][1]][0], trainerKeypoints[lines[i][1]][1])]
            pointToMerge.append(segment)

            segmentUser = [(userKeypoints[lines[i][0]][0], userKeypoints[lines[i][0]][1]),
                           (userKeypoints[lines[i][1]][0], userKeypoints[lines[i][1]][1])]
            pointToMergeUser.append(segmentUser)

        lc = mc.LineCollection(pointToMerge, colors='blue', linewidths=2)
        lcU = mc.LineCollection(pointToMergeUser, colors='green', linewidths=2)
        fig, ax = plt.subplots()
        ax.add_collection(lc)
        ax.add_collection(lcU)
        tr = mpatches.Patch(color='blue', label='Trainer')
        usr = mpatches.Patch(color='green', label='User')

        x = []
        xU = []
        y = []
        yU = []
        label = []
        for i in range(len(trainerKeypoints)):  # same length as len(lines)
            if plotHeadKeypoint is False and i in head: continue
            if plotFeetKeypoint is False and i in feet: continue
            label.append(i)
            x.append(trainerKeypoints[i][0])
            y.append(trainerKeypoints[i][1])

        for i in range(len(userKeypoints)):  # same length as len(lines)
            if plotHeadKeypoint is False and i in head: continue
            if plotFeetKeypoint is False and i in feet: continue
            xU.append(userKeypoints[i][0])
            yU.append(userKeypoints[i][1])

        title = 'KeyPoints'
        plt.title('Poses ' + str(trainerPose) + '-' + str(userPose) + title + '\n Video: ' + videoname)
        plt.scatter(x, y, color="white")
        plt.scatter(xU, yU, color="white")

        if plotKeypointsLabel is True:
            # the parameters (0.04, 0.1, ...) can be modified to obtain a better label spacing (difficult to find general valid parameters)
            count = 0
            trKp = mpatches.Patch(color='red', label='Trainer KeyPts')
            usrKp = mpatches.Patch(color='magenta', label='User KeyPts')
            for i in range(len(x)):
                if i in separate:
                    if count % 2 == 0:
                        plt.text(x[i] + 0.04, y[i] - 0.1, str(label[i]), color="red", fontsize=12)
                        plt.text(xU[i] + 0.04, yU[i] - 0.1, str(i), color="magenta", fontsize=12)
                        count += 1
                    else:
                        plt.text(x[i] - 0.09, y[i] + 0.2, str(label[i]), color="red", fontsize=12)
                        plt.text(xU[i] - 0.09, yU[i] + 0.2, str(label[i]), color="magenta", fontsize=12)
                        count += 1
                else:
                    plt.text(x[i], y[i] - 0.1, str(label[i]), color="red", fontsize=12)
                    plt.text(xU[i], yU[i] - 0.1, str(label[i]), color="magenta", fontsize=12)

        # invert the y axis to plot the pose in the same position as the person that do the exercise
        plt.gca().invert_yaxis()

        plt.legend(handles=[trKp, usrKp, tr, usr]) if plotKeypointsLabel is True else plt.legend(handles=[tr, usr])
        if countPose == 0:
            shutil.rmtree('posesImages/' + videoname, ignore_errors=True)
            os.makedirs('posesImages/' + videoname)
        figName = 'posesImages/' + videoname + '/pose' + str(trainerPose) + '-' + str(userPose) + '.png'
        plt.savefig(figName, dpi=600)
        # plt.savefig(figName)
        plt.close()
        if countPose % 25 == 0:
            if op is not None and opWrapper is not None:
                gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 2.5])
                plt.figure(figsize=(18, 8))

                img = plotKeyPointsImage(videoname, min + trainerPose, op, opWrapper)
                subfig = plt.subplot(gs[0, 0])
                subfig.set_title("Skeleton Pose " + str(trainerPose))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.axis('off')
                plt.imshow(img)

                img1 = plotKeyPointsImage(videoname, userMin + userPose, op, opWrapper, user=True)
                subfig1 = plt.subplot(gs[0, 1])
                subfig1.set_title("Skeleton User Pose " + str(userPose))
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                plt.axis('off')
                plt.imshow(img1)

                plt.subplot(gs[0, 2])
                img1 = cv2.imread(figName)
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                plt.axis('off')
                plt.imshow(img1)

                if wrongPosesMeanSTDIndex[couple] == -1:
                    plt.gcf().text(0.19, 0.1, 'Wrong for mean in std range checker', color="red", fontsize=15)
                else:
                    plt.gcf().text(0.19, 0.1, 'Correct for mean in std range checker', color="white", fontsize=15)

                if wrongPosesAngIndex is not None and wrongPosesAngIndex[couple] == -1:
                    plt.gcf().text(0.65, 0.1, 'Wrong for angles checker', color="red", fontsize=15)
                else:
                    plt.gcf().text(0.65, 0.1, 'Correct for angles checker', color="white", fontsize=15)

            else:
                img1 = cv2.imread(figName)
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                plt.axis('off')
                plt.imshow(img1)

                if wrongPosesMeanSTDIndex[couple] == -1:
                    plt.gcf().text(0.19, 0.1, 'Wrong for mean in std range checker', color="red", fontsize=15)
                else:
                    plt.gcf().text(0.19, 0.1, 'Correct for mean in std range checker', color="white", fontsize=15)

                if wrongPosesAngIndex is not None and wrongPosesAngIndex[couple] == -1:
                    plt.gcf().text(0.19, 0.03, 'Wrong for angles checker', color="red", fontsize=15)
                else:
                    plt.gcf().text(0.19, 0.03, 'Correct for angles checker', color="white", fontsize=15)

            plt.savefig(figName)
            plt.show()
            plt.close()
        countPose += 1


def plotPoseForAngles(pathAngles, videoname, min, userMin, op, opWrapper, wrongPosesMeanSTDIndex):
    countPose = 0
    plt.style.use('dark_background')
    for couple in range(len(pathAngles)):
        if countPose % 25 == 0:
            trainerPose = pathAngles[couple][0]
            userPose = pathAngles[couple][1]
            countPose += 1
            if countPose == 1:
                shutil.rmtree('posesImagesAngles/' + videoname, ignore_errors=True)
                os.makedirs('posesImagesAngles/' + videoname)

            gs = gridspec.GridSpec(1, 2)
            # if countPose % 10 == 0:  # one pose every 10 showed
            plt.figure(figsize=(18, 8))

            img = plotKeyPointsImage(videoname, min + trainerPose, op, opWrapper)
            subfig = plt.subplot(gs[0, 0])
            subfig.set_title("Skeleton Trainer Pose " + str(trainerPose))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.axis('off')
            plt.imshow(img)

            img1 = plotKeyPointsImage(videoname, userMin + userPose, op, opWrapper, user=True)
            subfig1 = plt.subplot(gs[0, 1])
            subfig1.set_title("Skeleton User Pose " + str(userPose))
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            plt.axis('off')
            plt.imshow(img1)

            if wrongPosesMeanSTDIndex[couple] == -1:
                plt.gcf().text(0.19, 0.1, 'Wrong for mean in std range checker', color="red", fontsize=15)
            else:
                plt.gcf().text(0.19, 0.1, 'Correct for mean in std range checker', color="white", fontsize=15)

            plt.savefig('posesImagesAngles/' + videoname + '/Poses ' + str(trainerPose) + '-' + str(userPose))
            plt.show()
            plt.close()
        countPose += 1


def printProgressBar(iteration, total, prefix='Progress:', suffix='', decimals=1, length=45, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    Print fancy progress bars. From Greenstick at StackOverflow
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    usage:
    for i in range(workamount):
        doWork(i)
        printProgressBar(i, workamount)
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
