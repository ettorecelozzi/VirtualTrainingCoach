from DevelopmentScripts import Statistics as stats
from DevelopmentScripts import PoseAnalysis
from DevelopmentScripts import Normalization as norm
import os
from DevelopmentScripts import PoseAlignment
import warnings
import pickle as pkl
from DevelopmentScripts.ExercisesParams import getPickle
import numpy as np
from DevelopmentScripts.PoseAnalysis import compareChecker


def main():
    videoname = 'SideStepsJacks'
    openposeT = False
    openposeU = False

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
    if openposeT is True:
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
    np.save('./KeyPoints/Trainer/' + videoname + '_normalized.npy', normalizedKeyPoints)

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
            getSkeletonPoints(videonameUser, 'User', op, opWrapper, True)
        else:
            getSkeletonPoints(videonameUser, 'User', op, opWrapper)
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
            alignedList = PoseAlignment.align1frame1pose(normalizedKeyPoints, mins, weights=None)

        # Get Trainer Statistics
        # pass True at the end of the method to use statistics library to calculate the std dev
        trainerMeans, stds = stats.removeDuplicateAndGetStat(alignedList, mins, normalizedKeyPoints)
        np.save('./KeypointsStatistics/Trainer/' + videoname + '_means.npy', trainerMeans)
        np.save('./KeypointsStatistics/Trainer/' + videoname + '_stds.npy', stds)

        #
        # ********************************* USER *********************************

        # Align User cycles
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            alignedUserList = PoseAlignment.align1frame1pose(normalizedUserKeyPoints, userMins, weights=None)

        # Get User statistics
        userMeans, userStds = stats.removeDuplicateAndGetStat(alignedUserList, userMins,
                                                              normalizedUserKeyPoints)
        np.save('./KeypointsStatistics/User/' + videoname + '_means.npy', userMeans)
        np.save('./KeypointsStatistics/User/' + videoname + '_stds.npy', userStds)

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

    ''' 
                   ************************************** ANGLES STRATEGY ************************************** 
    '''
    if switchcase == str(2) or switchcase == str(0):
        print('\n************************************** ANGLES STRATEGY **************************************')
        #
        # ********************************* TRAINER *********************************

        # Calculate and save joints angles
        keypoints = np.load('./KeyPoints/Trainer/' + videoname + '.npy')
        anglesKeyPoints, anglesKeyPointsSerialized = PoseAnalysis.getPointsAngles(keypoints, weights)
        np.save('./Angles/Trainer/' + videoname + '.npy', anglesKeyPoints)

        # Align trainer cycles
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            alignedListAngles = PoseAlignment.align1frame1pose(anglesKeyPointsSerialized, mins, weights, angles=True)

        # Get trainer statistics
        trainerMeansAngles, trainerStdsAngles = stats.getStatisticAngles(alignedListAngles, mins,
                                                                         anglesKeyPointsSerialized)

        #
        # ********************************* USER *********************************

        # Get User video keypoints
        userKeypoints = np.load('./KeyPoints/User/' + videonameUser + '.npy')

        # Retrieve and save joints angles
        anglesUserKeypoints, anglesUserKeypointsSerialized = PoseAnalysis.getPointsAngles(userKeypoints, weights)
        np.save('./Angles/User/' + videonameUser + '.npy', anglesUserKeypoints)

        # Align User cycles
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            alignedUserListAngles = PoseAlignment.align1frame1pose(anglesUserKeypointsSerialized, userMins, weights,
                                                                   angles=True)

        # Get User statistics
        userMeansAngles, userStdsAngles = stats.getStatisticAngles(alignedUserListAngles, userMins,
                                                                   anglesUserKeypointsSerialized)

        #
        # ********************************* ALIGN USER WITH THE TRAINER *********************************

        # User-trainer alignment
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            pathAngles = PoseAlignment.getDtwPath(trainerMeansAngles, userMeansAngles, angles=True)

        # Compare the trainer and user statistics through the "mean in std range" checker
        wrongPosesMeanSTDIndexAngles = compareChecker(trainerMeansAngles, userMeansAngles,
                                                      trainerStdsAngles, pathAngles,
                                                      weights=weights, errorStd=error, errorAllowed=10,
                                                      angles=True)


if __name__ == '__main__':
    main()
