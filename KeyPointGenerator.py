from DevelopmentScripts.KeyPointsFromVideo import getSkeletonPoints
from DevelopmentScripts.Openpose import opinit
from os import listdir

exerciseList = ['SideStepsJacks', 'SideStepsJacks-UserL']
print('Insert the name of the new video (in the Video folder), otherwise choose one video already available: \n',
      exerciseList)
videoname = input()
print('Is the video in landscape mode? Y/N')
rotate = input()
rotate = True if rotate.lower() == "n" else False

folder = 'User'

# OpenPose init
op, opWrapper = opinit()
exerciseName = ''
getSkeletonPoints(videoname, folder, exerciseName, op, opWrapper, rotate=True)

#
# f = 'Arm-clap'
# videoNames = listdir('./Videos/User/' + f)
# videoNames = [v.split('.')[0] for v in videoNames]
# for videoname in videoNames:
#     # Generate and save trainer skeleton keypoints
#     getSkeletonPoints(videoname, folder, f, op, opWrapper, rotate=True)
