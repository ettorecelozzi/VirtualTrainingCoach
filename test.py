from DevelopmentScripts import PoseAnalysis
import numpy as np
from DevelopmentScripts import Normalization as norm
from DevelopmentScripts import PoseAlignment
from DevelopmentScripts import Statistics as stats
from DevelopmentScripts import Utility
import pickle as pkl
from DevelopmentScripts.ExercisesParams import getPickle
from DevelopmentScripts.PoseAnalysis import *

# kps_kth = np.load('./KTH-Joints/kth_output_NPY/person01_boxing_d1.npy')
# plt = Utility.plotPoseFromKeypoints(kps_kth[0])
# plt.show()

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


wrongPoses, wrongPosesIndex = compareChecker(trainerCycle=cycleTrainer, userCycle=meansUser, path=path, weights=weights,
                                             errorAngles=15)

# plot the trainer vs means user poses
Utility.plotTrainerVsUser(path=path, wrongPoses=wrongPosesIndex, keypoints=cycleTrainer, keypointsUser=meansUser,
                          videonameTrainer=cleanName)

wrongPosesIndexGood = [0 if i == -1 else -1 for i in wrongPosesIndex]
Utility.plotTrainerVsUser(path=path, wrongPoses=wrongPosesIndexGood, keypoints=cycleTrainer, keypointsUser=meansUser,
                          videonameTrainer=cleanName + '_Good')

# plot the trainer vs best and worst user cycle poses
getBestWorstCyclePoses(videoname=cleanName, wrongPoses=wrongPoses, alignedListUser=alignedListUser,
                       cycleTrainer=cycleTrainer, normalizedKeyPointsUser=normalizedKeyPointsUser, minsUser=minsUser,
                       weights=weights)
