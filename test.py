from DevelopmentScripts import PoseAnalysis
import numpy as np
from DevelopmentScripts import Normalization as norm
from DevelopmentScripts import PoseAlignment
from DevelopmentScripts import Statistics as stats
from DevelopmentScripts import Utility
import pickle as pkl
from DevelopmentScripts.ExercisesParams import getPickle

getPickle()
videoname = 'arm-clap'
clenaName = Utility.getCleanName(videoname, False)[0]
params = pkl.load(open('paramsPickle/' + clenaName, 'rb'))
weights = params[0]
slidingWindowDim = params[1]['slidingWindowDimension']

keyPoints = np.load('./KeyPoints/Trainer/' + videoname + '.npy')
meanTorso, meanHipX, meanHipY = PoseAnalysis.getMeanMeasures(keyPoints, 50)
normalizedKeyPointsTrainer = norm.normalize(meanTorso, meanHipX, meanHipY, keyPoints.copy())
print("before: " + str(len(normalizedKeyPointsTrainer)))
normalizedKeyPointsTrainer = PoseAlignment.deleteEqualPoses(normalizedKeyPointsTrainer)
print("after: " + str(len(normalizedKeyPointsTrainer)))
minsTrainer = PoseAnalysis.extractCyclesByEuclidean(slidingWindowDim, normalizedKeyPointsTrainer, weights=weights,
                                                    plotChart=True)
print(minsTrainer)

# videoname = 'double-lunges'
keyPointsUser = np.load('./KeyPoints/User/' + videoname + '_1.npy')
normalizedKeyPointsUser = norm.normalize(meanTorso, meanHipX, meanHipY, keyPointsUser.copy())
print("before: " + str(len(normalizedKeyPointsUser)))
normalizedKeyPointsUser = PoseAlignment.deleteEqualPoses(normalizedKeyPointsUser)
print("after: " + str(len(normalizedKeyPointsUser)))
minsUser = PoseAnalysis.extractCyclesByEuclidean(slidingWindowDim, normalizedKeyPointsUser, weights=weights,
                                                 plotChart=True,
                                                 sequence1=normalizedKeyPointsTrainer[0:slidingWindowDim])
print(minsUser)

alignedList = PoseAlignment.align1frame1pose(normalizedKeyPointsTrainer, minsTrainer, weights=None)
meansTrainer, stdsTrainer = stats.getStats(alignedList, minsTrainer, normalizedKeyPointsTrainer)
alignedListUser = PoseAlignment.align1frame1pose(normalizedKeyPointsUser, minsUser, weights=None)
meansUser, stdsUser = stats.getStats(alignedListUser, minsUser, normalizedKeyPointsUser)
path = PoseAlignment.getDtwPath(meansTrainer, meansUser)
Utility.plotIndexOfFit(path, stdsUser, stdsTrainer)
