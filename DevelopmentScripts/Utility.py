import matplotlib.pyplot as plt
from matplotlib import collections as mc
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import os
import numpy as np
from DevelopmentScripts.KeyPointsFromImage import plotKeyPointsImage
import cv2
import shutil
from os import listdir
import difflib


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


def getCleanName(videoname, user=False):
    """
    Given the name of a video, the closest name in the db is returned.
    To notice: doesn't scale well with the number of exercise name to check
    :param videoname: string
    :param user: boolean. User video or not
    :return: string. closest exercise name
    """
    filesNames = listdir('./paramsPickle')
    filesNames = [f.lower() for f in filesNames]
    cleanName = difflib.get_close_matches(videoname.lower(), filesNames)[0]  # return always the closest match

    videoNames = listdir('./Videos/Trainer') if user is False else listdir('./Videos/User')
    videoNames = [v.split('.')[0].lower() for v in videoNames]
    noErrorName = difflib.get_close_matches(videoname.lower(), videoNames)[0]

    return cleanName, noErrorName


def plotPoseFromKeypoints(keypoints, ax, lbl='Trainer', color=False):
    """
    Plot poses given keypoints
    :param keypoints: one frame, (25,2)
    :param ax: axis where to plot
    :param label: user or trainer keypoints
    :param color: decide to change or not the color
    :return: plt object
    """
    colors = ['blue', 'white', 'red'] if color is False else ['green', 'white', 'red']
    # coordinates of the points to tie
    lines = [(0, 15), (0, 16), (0, 17), (0, 18), (0, 1), (1, 8), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (8, 9),
             (9, 10), (10, 11), (11, 24), (11, 22), (23, 22), (22, 24), (8, 12), (12, 13), (13, 14), (14, 19), (19, 20),
             (21, 19), (21, 14)]
    # index of the label to space
    separate = [11, 24, 23, 22, 14, 21, 19, 20]
    head = [15, 16, 17, 18]
    feet = [24, 23, 22, 21, 19, 20]
    pointToMerge = []
    for i in range(len(lines)):
        if lines[i][0] in head or lines[i][1] in head: continue
        if lines[i][0] in feet or lines[i][1] in feet: continue
        segment = [(keypoints[lines[i][0]][0], keypoints[lines[i][0]][1]),  # (x1,y1) (x2,y2) segment
                   (keypoints[lines[i][1]][0], keypoints[lines[i][1]][1])]
        pointToMerge.append(segment)

    lc = mc.LineCollection(pointToMerge, colors=colors[0], linewidths=2)
    ax.add_collection(lc)
    kp = mpatches.Patch(color=colors[0], label=lbl)
    x = []
    y = []
    label = []
    for i in range(len(keypoints)):  # same length as len(lines)
        if i in head: continue
        if i in feet: continue
        label.append(i)
        x.append(keypoints[i][0])
        y.append(keypoints[i][1])

    plt.scatter(x, y, color=colors[1])
    # plt.scatter(400, 300, color="black")  # values needed to avoid that the pose is stretched
    # plt.scatter(800, 500, color="black")
    count = 0
    Kp_L = mpatches.Patch(color=colors[2], label=lbl + ' KeyPts')
    for i in range(len(x)):
        if i in separate:
            if count % 2 == 0:
                plt.text(x[i] + 0.04, y[i] - 0.2, str(label[i]), color=colors[2], fontsize=12)
                count += 1
            else:
                plt.text(x[i] - 0.09, y[i] + 0.09, str(label[i]), color=colors[2], fontsize=12)
                count += 1
        else:
            plt.text(x[i], y[i] - 0.1, str(label[i]), color=colors[2], fontsize=12)

    # invert the y axis to plot the pose in the same position as the person that do the exercise
    plt.gca().invert_yaxis()
    if color is True:
        usr = mpatches.Patch(color=colors[0], label='User')
        kp = mpatches.Patch(color="blue", label='Trainer')
        plt.legend(handles=[kp, usr])
        return plt
    plt.legend(handles=[Kp_L, kp])
    return plt


def plotFromAlignedList(alignedList, mins, keypoints, videoname):
    """
    Given alignedList plot all the poses in a list
    :param alignedList: list of aligned poses
    :param mins: cycles bounds
    :param keypoints: keypoints to plot (25,2)
    :param videoname: string
    """
    plt.style.use('dark_background')
    for list in alignedList:
        numOfFigures = len(list) - 1
        columns = 3
        rows = int(np.ceil(numOfFigures / 3))
        # columns = int(np.ceil(numOfFigures / 3)) if int(np.ceil(numOfFigures / 3)) >= 2 else: 2
        mainFig = plt.figure(figsize=(18, 8))
        mainFig.suptitle(videoname)
        gs = mainFig.add_gridspec(rows, columns)
        row, column = 0, 0
        for el in list:
            pose, cycle = el.split("|")
            pose, cycle = int(pose), int(cycle)
            ax = mainFig.add_subplot(gs[row, column])
            plotPoseFromKeypoints(keypoints[mins[cycle] + pose], mins[cycle] + pose, ax)
            ax.set_title("Pose " + str(mins[cycle] + pose))
            if column >= columns - 1:
                column = 0
                row += 1
            else:
                column += 1
    plt.show()


def plotTrainerVsUser(path, wrongPoses, keypoints, keypointsUser, videonameTrainer, min=0, userMin=0,
                      videonameUser=None, op=None, opWrapper=None):
    """
    Given the last path that compare the trainer exercise with the user and plot the poses.
    :param path: aligned poses
    :param wrongPoses: indexes of the poses wrong
    :param keypoints: keypoints, (25,2)
    :param keypointsUser: user keypoints, (25,2)
    :param videonameTrainer: string
    :param min: min
    :param userMin: user min
    :param videonameUser: string
    :param op: Openpose tool
    :param opWrapper: Openpose tool
    """
    if not os.path.exists('./Plots/TrainerVsUser/' + videonameTrainer):
        os.makedirs('./Plots/TrainerVsUser/' + videonameTrainer)

    plt.style.use('dark_background')
    videonameUser = videonameTrainer if videonameUser is None else videonameUser
    userMin = min if userMin is None else userMin
    countPose = 0
    for couple in range(len(path)):
        trainerPose = path[couple][0]
        userPose = path[couple][1]
        trainerKeypoints = keypoints[trainerPose]  # frame keypoints for the trainerPose
        userKeypoints = keypointsUser[userPose]

        if wrongPoses[couple] == -1:  # change here to choose which poses show
            mainFig = plt.figure(figsize=(18, 8))
            mainFig.suptitle(videonameTrainer)
            gs = mainFig.add_gridspec(1, 3)
            if op is not None and opWrapper is not None:
                img = plotKeyPointsImage(videonameTrainer, min + trainerPose, op, opWrapper)
                subfig = mainFig.add_subplot(gs[0, 0])
                subfig.set_title("Skeleton Pose " + str(trainerPose))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.axis('off')
                plt.imshow(img)

                img1 = plotKeyPointsImage(videonameUser, userMin + userPose, op, opWrapper, user=True)
                subfig1 = mainFig.add_subplot(gs[0, 1])
                subfig1.set_title("Skeleton User Pose " + str(userPose))
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                plt.axis('off')
                plt.imshow(img1)

            else:
                subfig = mainFig.add_subplot(gs[0, 0])
                subfig.set_title("Skeleton Pose " + str(trainerPose))
                plotPoseFromKeypoints(trainerKeypoints, subfig, lbl='Trainer')

                subfig1 = mainFig.add_subplot(gs[0, 1])
                subfig1.set_title("Skeleton User Pose " + str(userPose))
                plotPoseFromKeypoints(userKeypoints, subfig1, lbl='User')

            subfig2 = mainFig.add_subplot(gs[0, 2])
            subfig2.set_title("Poses overlay")
            plotPoseFromKeypoints(trainerKeypoints, subfig2)
            plotPoseFromKeypoints(userKeypoints, subfig2, color=True)
            plt.gca().invert_yaxis()

            if wrongPoses[couple] == -1:
                plt.gcf().text(0.45, 0.04, 'Wrong for angles checker', color="red", fontsize=15)
            else:
                plt.gcf().text(0.45, 0.04, 'Correct for angles checker', color="white", fontsize=15)
            # plt.show()
            plt.savefig('./Plots/TrainerVsUser/' + videonameTrainer + '/Trainer ' + str(min + trainerPose) +
                        ' User ' + str(userMin + userPose) + '.png')
            plt.close()
        countPose += 1


def plotIndexOfFit(path, stdsUser, stdsTrainer, firstCycle=False, title=None):
    trainerM = []
    userM = []
    for couple in path:
        # trainerM.append(np.mean(stdsTrainer[couple[0]].flatten()))
        tmpt = []
        tmpu = []
        for sublist in stdsTrainer[couple[0]]:
            for el in sublist:
                tmpt.append(el)
        for sublist in stdsUser[couple[1]]:
            for el in sublist:
                tmpu.append(el)
        trainerM.append(sum(tmpt) / len(tmpt))
        userM.append(sum(tmpu) / len(tmpu))
        # fitIndex.append((1/trainermean)/(1/usermean))
    # trainerWeights = 1/trainerM
    for i in range(0, len(trainerM)):
        trainerM[i] = 1 / trainerM[i]
    for i in range(0, len(userM)):
        userM[i] = 1 / userM[i]
    fits = []
    for i in range(0, len(trainerM)):
        fits.append(userM[i] / trainerM[i])
    plt.plot(userM, label='User')
    plt.plot(trainerM, label='Training coach')
    plt.plot(fits, label='Index of fit')
    plt.legend(loc='upper left')
    plt.title(title)
    # plt.savefig('./Plots/IndexOfFit/First cycle/' + title) if firstCycle is True else plt.savefig(
    #     './Plots/IndexOfFit/Normal/' + title)
    plt.show()


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
