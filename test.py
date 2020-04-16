from DevelopmentScripts import PoseAnalysis
import numpy as np
from DevelopmentScripts import Normalization as norm
from DevelopmentScripts import PoseAlignment
from DevelopmentScripts import Statistics as stats
from DevelopmentScripts import Utility
import pickle as pkl
from DevelopmentScripts.ExercisesParams import getPickle
from DevelopmentScripts.PoseAnalysis import *

getPickle()
videoname = 'arm-clap'
cleanName = Utility.getCleanName(videoname, False)[0]
params = pkl.load(open('paramsPickle/' + cleanName, 'rb'))
weights = params[0]
slidingWindowDim = params[1]['slidingWindowDimension']

keyPoints = np.load('./KeyPoints/Trainer/' + videoname + '.npy')
getPointsAngles(keyPoints, weights)
meanTorso, meanHipX, meanHipY = PoseAnalysis.getMeanMeasures(keyPoints, 50)
normalizedKeyPointsTrainer = norm.normalize(meanTorso, meanHipX, meanHipY, keyPoints.copy())
# print("before: " + str(len(normalizedKeyPointsTrainer)))
normalizedKeyPointsTrainer = PoseAlignment.deleteEqualPoses(normalizedKeyPointsTrainer)
# print("after: " + str(len(normalizedKeyPointsTrainer)))
minsTrainer = PoseAnalysis.extractCyclesByEuclidean(slidingWindowDim, normalizedKeyPointsTrainer, weights=weights,
                                                    plotChart=False)
print(minsTrainer)

# videoname = 'double-lunges'
keyPointsUser = np.load('./KeyPoints/User/' + videoname + '_1.npy')
normalizedKeyPointsUser = norm.normalize(meanTorso, meanHipX, meanHipY, keyPointsUser.copy())
# print("before: " + str(len(normalizedKeyPointsUser)))
normalizedKeyPointsUser = PoseAlignment.deleteEqualPoses(normalizedKeyPointsUser)
# print("after: " + str(len(normalizedKeyPointsUser)))
minsUser = PoseAnalysis.extractCyclesByEuclidean(slidingWindowDim, normalizedKeyPointsUser, weights=weights,
                                                 plotChart=False,
                                                 sequence1=normalizedKeyPointsTrainer[0:slidingWindowDim])
print(minsUser)

# alignedListTrainer = PoseAlignment.align1frame1pose(normalizedKeyPointsTrainer, minsTrainer, weights=None)
# meansTrainer, stdsTrainer = stats.getStats(alignedListTrainer, minsTrainer, normalizedKeyPointsTrainer)
cycleTrainer = normalizedKeyPointsTrainer[minsTrainer[0]:minsTrainer[1]]
alignedListUser = PoseAlignment.align1frame1pose(normalizedKeyPointsUser, minsUser, weights=None)
meansUser, stdsUser = stats.getStats(alignedListUser, minsUser, normalizedKeyPointsUser)
path = PoseAlignment.getDtwPath(cycleTrainer, meansUser)
# Utility.plotIndexOfFit(path, stdsUser, stdsTrainer)

# wrongPosesAngles = compareChecker(trainerCycle=cycleTrainer, userCycle=meansUser, path=path, weights=weights,
#                                   errorAngles=15)
# Utility.plotTrainerVsUser(path=path, wrongPoses=wrongPosesAngles, keypoints=cycleTrainer, keypointsUser=meansUser,
#                           videonameTrainer=cleanName)

wrongPoses = compareChecker(trainerCycle=cycleTrainer, userCycle=meansUser, path=path, weights=weights,
            errorAngles=15)
for el in wrongPoses:
    wrongMeanPoses = alignedListUser[el[1]] #this is the list of poses that belongs to the wrong mean pose
    #now we should evaluate each pose of the mean pose and get the worst and the best
    p = []
    d = []
    for pose in wrongMeanPoses:
        idx,cycle = pose.split("|")
        poseTrainer = cycleTrainer[el[0]]
        poseUser = normalizedKeyPointsUser[int(idx)+minsUser[int(cycle)]]
        distance = getAngleDistance(trainerPose=poseTrainer,userPose=poseUser, errorAngles=15, weights=weights)
        p.append(pose)
        d.append(distance)
    maxd = max(d)
    mind = min(d)
    bestp = p[d.index(mind)]
    worstp = p[d.index(maxd)]
    print("for pose "+str(el[1])+" the best is "+bestp+" with distance "+str(mind)+" and the worst is "+worstp+" with distance "+str(maxd))