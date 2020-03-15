from DevelopmentScripts import Statistics as stats
from DevelopmentScripts import PoseAnalysis
from DevelopmentScripts import Normalization as norm
from DevelopmentScripts import PoseAlignment
import warnings
import pickle as pkl
from DevelopmentScripts.ExercisesParams import getPickle
from DevelopmentScripts.Utility import *
from DevelopmentScripts.PoseAnalysis import compareChecker


def main():
    videoname = 'SideSteJacks'
    openposeT = False
    openposeU = False

    # params
    getPickle()
    exerciseParams, cleanName = getCleanName(videoname)
    print('Exercise name: ' + exerciseParams)

    videonameUser = cleanName + '-UserL'
    params = pkl.load(open('paramsPickle/' + cleanName, 'rb'))

    slidingWindowDimension = params[1]['slidingWindowDimension']
    slidingWindowDimensionUser = params[1]['slidingWindowDimensionUser']
    meanRange = params[1]['meanRange']  # represents the interval [-meanRange; +meanRange]
    firstMin_TrainerCycle = params[1]['firstMin_TrainerCycle']  # first mins' index
    error = params[1]['error']

    print('\nExercise params: ', params[1])
    weights = params[0]

    #
    # ********************************* TRAINER *********************************

    # Generate and save trainer skeleton keypoints
    if openposeT is True:
        from DevelopmentScripts.KeyPointsFromVideo import getSkeletonPoints
        from DevelopmentScripts.Openpose import opinit

        # OpenPose init
        op, opWrapper = opinit()

        getSkeletonPoints(cleanName, 'Trainer', cleanName, op, opWrapper)
        opWrapper.stop()

    # Normalize (and save) trainer keypoints through mean value retrieved by the meanRange
    keyPoints = np.load('./KeyPoints/Trainer/' + cleanName + '.npy')
    meanTorso, meanHipX, meanHipY = PoseAnalysis.getMeanMeasures(keyPoints, meanRange)
    normalizedKeyPoints = norm.normalize(meanTorso, meanHipX, meanHipY, keyPoints.copy())
    np.save('./KeyPoints/Trainer/' + cleanName + '_normalized.npy', normalizedKeyPoints)

    # Extract trainer cycles
    mins = PoseAnalysis.extractCyclesByDtw(slidingWindowDimension, keyPoints, plotChart=False)
    print('\nIndexes of the Trainer cycles: ', mins)

    #
    # ********************************* USER *********************************

    # Generate and save User skeleton keypoints
    if openposeU is True:
        from DevelopmentScripts.KeyPointsFromVideo import getSkeletonPoints
        from DevelopmentScripts.Openpose import opinit
        # OpenPose init
        op, opWrapper = opinit()
        rotate = input("\nDo you want to rotate the video? Y/N ")
        if rotate.lower() == 'y':
            getSkeletonPoints(videonameUser, 'User', cleanName, op, opWrapper, True)
        else:
            getSkeletonPoints(videonameUser, 'User', cleanName, op, opWrapper)
        opWrapper.stop()

    # Normalize (and save) User keypoints through mean value retrieved by the meanRange
    userKeypoints = np.load('./KeyPoints/User/' + videonameUser + '.npy')
    userMeanTorso, userMeanHipX, userMeanHipY = PoseAnalysis.getMeanMeasures(userKeypoints, meanRange)
    normalizedUserKeyPoints = norm.normalize(userMeanTorso, userMeanHipX, userMeanHipY, userKeypoints.copy())
    np.save('./KeyPoints/User/' + videonameUser + '_normalized.npy', normalizedKeyPoints)

    # Define the template (trainerCycle) and extract User cycles
    trainerCycle = normalizedKeyPoints[
                   mins[firstMin_TrainerCycle]: (mins[firstMin_TrainerCycle] + slidingWindowDimension)]
    userMins = PoseAnalysis.extractCyclesByDtw(slidingWindowDimensionUser, normalizedUserKeyPoints, plotChart=False,
                                               sequence1=trainerCycle, user=True)
    print('\nIndexes of the User cycles: ', userMins)

    print('\n************************************** X-Y KEYPOINTS STRATEGY **************************************')
    #
    # ********************************* TRAINER *********************************

    # Align trainer cycles
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        alignedList = PoseAlignment.align1frame1pose(normalizedKeyPoints, mins, weights=None)

    plotFromAlignedList(alignedList, mins, normalizedKeyPoints, cleanName)

    # Get Trainer Statistics
    # pass True at the end of the method to use statistics library to calculate the std dev
    trainerMeans, stds = stats.removeDuplicateAndGetStat(alignedList, mins, normalizedKeyPoints)
    np.save('./KeypointsStatistics/Trainer/' + cleanName + '_means.npy', trainerMeans)
    np.save('./KeypointsStatistics/Trainer/' + cleanName + '_stds.npy', stds)

    #
    # ********************************* USER *********************************

    # Align User cycles
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        alignedUserList = PoseAlignment.align1frame1pose(normalizedUserKeyPoints, userMins, weights=None)

    # Get User statistics
    userMeans, userStds = stats.removeDuplicateAndGetStat(alignedUserList, userMins,
                                                          normalizedUserKeyPoints)
    np.save('./KeypointsStatistics/User/' + videonameUser + '_means.npy', userMeans)
    np.save('./KeypointsStatistics/User/' + videonameUser + '_stds.npy', userStds)

    #
    # ********************************* ALIGN USER WITH THE TRAINER *********************************

    # User-trainer alignment
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        path = PoseAlignment.getDtwPath(trainerMeans, userMeans)
    #
    # ********************************* COMPARE USER TRAINER EXERCISE *********************************

    # Compare the trainer and User statistics through the two checker
    wrongPosesMeanSTDIndex = compareChecker(trainerMeans, userMeans, stds, path,
                                            weights=weights, errorStd=error,
                                            errorAllowed=10)


if __name__ == '__main__':
    main()
