from DevelopmentScripts import Statistics as stats
from DevelopmentScripts import PoseAnalysis
from DevelopmentScripts import Normalization as norm
import os
from DevelopmentScripts import PoseAlignment
import warnings
import pickle as pkl
from DevelopmentScripts.ExercisesParams import getPickle
import numpy as np
from DevelopmentScripts.Utility import compareChecker


def main():
    videoname = 'SideStepsJacks'
    openpose = False

    # params
    getPickle()

    videonameUser = videoname + '-UserL'
    params = pkl.load(open('paramsPickle/' + videoname, 'rb'))

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
    if openpose is True:
        from DevelopmentScripts.KeyPointsFromVideo import getSkeletonPoints
        from DevelopmentScripts.Openpose import opinit

        # OpenPose init
        op, opWrapper = opinit()

        getSkeletonPoints(videoname, 'Trainer', op, opWrapper)
        opWrapper.stop()

    # Normalize (and save) trainer keypoints through mean value retrieved by the meanRange
    keyPoints = np.load('./KeyPoints/Trainer/' + videoname + '.npy')
    meanTorso, meanHipX, meanHipY = PoseAnalysis.getMeanMeasures(keyPoints, meanRange)
    normalizedKeyPoints = norm.normalize(meanTorso, meanHipX, meanHipY, keyPoints.copy())
    np.save('./KeyPoints/' + videoname + '_normalized.npy', normalizedKeyPoints)

    # Extract trainer cycles
    mins = PoseAnalysis.extractCyclesByDtw(slidingWindowDimension, keyPoints, True)
    print('\nIndexes of the Trainer cycles: ', mins)

    #
    # ********************************* USER *********************************

    # Generate and save user skeleton keypoints
    if openpose is True:
        from DevelopmentScripts.KeyPointsFromVideo import getSkeletonPoints
        from DevelopmentScripts.Openpose import opinit
        # OpenPose init
        op, opWrapper = opinit()
        rotate = input("\nDo you want to rotate the video? Y/N ")
        if rotate.lower() == 'y':
            getSkeletonPoints(videonameUser, 'User', op, opWrapper, True)
        else:
            getSkeletonPoints(videonameUser, 'User', op, opWrapper)
        opWrapper.stop()

    # Normalize (and save) user keypoints through mean value retrieved by the meanRange
    userKeypoints = np.load('./KeyPoints/User/' + videonameUser + '.npy')
    userMeanTorso, userMeanHipX, userMeanHipY = PoseAnalysis.getMeanMeasures(userKeypoints, meanRange)
    normalizedUserKeyPoints = norm.normalize(userMeanTorso, userMeanHipX, userMeanHipY, userKeypoints.copy())
    np.save('./KeyPoints/' + videonameUser + '_normalized.npy', normalizedKeyPoints)

    # Define the template (trainerCycle) and extract user cycles
    trainerCycle = normalizedKeyPoints[
                   mins[firstMin_TrainerCycle]: (mins[firstMin_TrainerCycle] + slidingWindowDimension)]
    userMins = PoseAnalysis.extractCyclesByDtw(slidingWindowDimensionUser, normalizedUserKeyPoints, True,
                                               sequence1=trainerCycle, user=True)
    print('\nIndexes of the User cycles: ', userMins)

    print("\nChoose the strategy:\n 1 X-Y Keypoints Strategy \n 2 Angles Strategy \n 0 to execute both")
    switchcase = input()

    ''' 

            ************************************** X-Y KEYPOINTS STRATEGY ************************************** 

    '''
    if switchcase == str(1) or switchcase == str(0):
        print('\n************************************** X-Y KEYPOINTS STRATEGY **************************************')
        #
        # ********************************* TRAINER *********************************

        # Align trainer cycles
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            alignedList = PoseAlignment.align1frame1pose(normalizedKeyPoints, mins, weights)

        # Get Trainer Statistics
        # pass True at the end of the method to use statistics library to calculate the std dev
        trainerMeans, stds = PoseAlignment.removeDuplicateAndGetStat(alignedList, mins, normalizedKeyPoints)
        np.save('./KeyPointsStatistics/Trainer/' + videoname + '_means.npy', trainerMeans)
        np.save('./KeyPointsStatistics/Trainer/' + videoname + '_stds.npy', stds)

        #
        # ********************************* USER *********************************

        # Align user cycles
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            alignedUserList = PoseAlignment.align1frame1pose(normalizedUserKeyPoints, userMins, weights)

        # Get User statistics
        userMeans, userStds = PoseAlignment.removeDuplicateAndGetStat(alignedUserList, userMins,
                                                                      normalizedUserKeyPoints)
        np.save('./KeyPointsStatistics/User/' + videoname + '_means.npy', userMeans)
        np.save('./KeyPointsStatistics/User/' + videoname + '_stds.npy', userStds)

        #
        # ********************************* ALIGN USER WITH THE TRAINER *********************************

        # User-trainer alignment
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            path = PoseAlignment.getDtwPath(trainerMeans, userMeans, weights)
        #
        # ********************************* COMPARE USER TRAINER EXERCISE *********************************

        # Compare the trainer and user statistics through the two checker
        wrongPosesMeanSTDIndex, wrongPosesAngIndex = compareChecker(trainerMeans, userMeans, stds, path,
                                                                    errorAllowed=10)


if __name__ == '__main__':
    main()
