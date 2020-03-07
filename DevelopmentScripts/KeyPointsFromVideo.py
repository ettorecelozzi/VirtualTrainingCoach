import cv2
import os
import shutil
import numpy as np


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    :param mat: matrix
    :param angle: angle of rotation
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
        width / 2,
        height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def getSkeletonPoints(videoname, folder, op, opWrapper, rotate=False):
    """
    Retrieve the skeleton from a video frame by frame
    :param videoname: string
    :param folder: string (folder name)
    :param op: Openpose tool from initialization
    :param opWrapper: Openpose tool from initialization
    :param rotate: boolean to specify if a rotation is needed
    :return: boolean, extraction successful or not
    """
    cam = cv2.VideoCapture('./Videos/' + videoname + '.mp4')

    # creating frames' folder
    if os.path.exists('Frames/' + videoname):
        shutil.rmtree('./Frames/' + videoname + '/')
    os.makedirs('Frames/' + videoname)

    # frame
    currentframe = -1  # starting frame (increment done before the iteration)
    frameFrequency = 0
    keypoints = []
    while True:

        # reading from frame
        ret, frame = cam.read()
        if ret:
            currentframe += 1
            if frameFrequency == 0 or currentframe % frameFrequency == 0:
                datum = op.Datum()

                name = './Frames/' + videoname + '/frame' + str(currentframe) + '.jpg'
                print('Creating...' + name)
                # writing the extracted images
                if rotate is True:
                    frame = rotate_image(frame, 270)
                cv2.imwrite(name, frame)

                # process frame
                imgToProcess = cv2.imread(name)
                datum.cvInputData = imgToProcess
                opWrapper.emplaceAndPop([datum])

                # Display Image
                # print("Body keypoints: \n" + str(datum.poseKeypoints))
                img = datum.cvOutputData
                # cv2.imshow("frame" + str(currentframe), img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                if datum.poseKeypoints.size > 1:
                    keypoints.append(datum.poseKeypoints[0][:, :2])  # list append is faster than numpy append
                else:
                    keypoints.append(np.zeros((25, 2)))
        else:
            if currentframe == -1:
                print("Video not found")
                return
            else:
                keypoints = np.asarray(keypoints)  # numpy arrays are more handy and memory efficient
                np.save('./KeyPoints/' + folder + '/' + videoname + '.npy', keypoints)
                print("\nVideo ended\n")
                # Release all space and windows once done
                cam.release()
                cv2.destroyAllWindows()
                return True
