# It requires OpenCV installed for Python
import cv2


def plotKeyPointsImage(videoname, frameNumber, op, opWrapper, user=False):  # frame and pose number is the same
    """
    Given an image, use Openpose to retrieve the skeleton
    :param videoname: string
    :param frameNumber: int. Index in the frames folder
    :param op: Openpose tool from initialization
    :param opWrapper: Openpose tool from initialization
    :param user: boolean to specify if the video is for the User
    :return: output skeleton
    """
    if user is False:
        imgname = 'Frames/' + videoname.split('-')[0] + '/frame' + str(frameNumber) + '.jpg'
    else:
        imgname = 'Frames/' + videoname + '/frame' + str(frameNumber) + '.jpg'
    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(imgname)
    # imageToProcess = rotate_image(imageToProcess, 270) # anticlockwise orientation
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])
    return datum.cvOutputData
