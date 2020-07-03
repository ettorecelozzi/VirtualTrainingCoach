from DevelopmentScripts.KeyPointsFromVideo import getSkeletonPoints
from DevelopmentScripts.Openpose import opinit
from os import listdir
from DevelopmentScripts.KeyPointsFromImage import plotKeyPointsImage
import numpy as np

# exerciseList = ['SideStepsJacks', 'SideStepsJacks-UserL']
# print('Insert the name of the new video (in the Video folder), otherwise choose one video already available: \n',
#       exerciseList)
# videoname = input()
# print('Is the video in landscape mode? Y/N')
# rotate = input()
# rotate = True if rotate.lower() == "n" else False


# OpenPose init
op, opWrapper = opinit()

# exerciseName = ''
# getSkeletonPoints(videoname, folder, exerciseName, op, opWrapper, rotate=True)

# for file in listdir('./Yoga/Yoga Poses'):
#     kps = plotKeyPointsImage('./Yoga/Yoga Poses/' + file, None, op, opWrapper)[1]
#     ex_name = file.split('.')[0]
#     np.save('./Yoga/Yoga Keypoints/' + ex_name + '.npy', kps[:, :2])

folder = 'Squat/'
f = ''
videoNames = listdir('./Dataset/Videos/' + folder)
for videoname in videoNames:
    # Generate and save trainer skeleton keypoints
    videoname = videoname[:len(videoname) - 4]
    getSkeletonPoints(videoname, folder, f, op, opWrapper, rotate=False)
