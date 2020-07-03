import cv2
import os
import shutil
import numpy as np
import subprocess


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


def getSkeletonPoints(videoname, folder, exerciseName, op, opWrapper, rotate=False):
    """
    Retrieve the skeleton from a video frame by frame
    :param videoname: string
    :param folder: string (folder, Trainer or User)
    :param op: Openpose tool from initialization
    :param opWrapper: Openpose tool from initialization
    :param rotate: boolean to specify if a rotation is needed
    :return: boolean, extraction successful or not
    """
    video_input_path = './Dataset/Videos/' + folder + '/' + videoname + '.mp4'
    video_output_path = './Dataset/Videos/' + folder + '/' + videoname + '_30fps.mp4'
    c = 'ffmpeg -y -i ' + video_input_path + ' -r 30 -c:v libx264 -b:v 3M -strict -2 -movflags faststart ' \
        + video_output_path
    subprocess.call(c, shell=True)
    os.remove(video_input_path)
    os.rename(video_output_path, video_input_path)

    cam = cv2.VideoCapture('./Dataset/Videos/' + folder + '/' + videoname + '.mp4')
    print('Video fps: ' + str(cam.get(cv2.CAP_PROP_FPS)))

    # creating frames' folder
    if not os.path.exists('./Dataset/Frames/' + videoname):
        os.makedirs('./Dataset/Frames/' + videoname)

    # frame
    currentframe = -1  # starting frame (increment done before the iteration)
    frameFrequency = 0
    keypoints = []
    keypoints_flipped = []
    keypoints_withConfidence = []
    keypoints_withConfidence_flipped = []
    while True:

        # reading from frame
        ret, frame = cam.read()
        if ret:
            currentframe += 1
            if frameFrequency == 0 or currentframe % frameFrequency == 0:
                datum = op.Datum()

                name = './Dataset/Frames/' + videoname + '/frame' + str(currentframe) + '.jpg'
                print('Creating...' + name)
                # writing the extracted images
                if rotate is True:
                    frame = rotate_image(frame, 270)
                cv2.imwrite(name, frame)

                # process frame
                imgToProcess = cv2.imread(name)
                datum.cvInputData = imgToProcess
                opWrapper.emplaceAndPop([datum])
                # os.remove(name)

                # Display Image
                # print("Body keypoints: \n" + str(datum.poseKeypoints))
                img = datum.cvOutputData
                # cv2.imshow("frame" + str(currentframe), img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                if datum.poseKeypoints.size > 1:
                    keypoints.append(datum.poseKeypoints[0][:, :2])  # list append is faster than numpy append
                    keypoints_withConfidence.append(datum.poseKeypoints[0])
                else:
                    keypoints.append(np.zeros((25, 2)))
                    keypoints_withConfidence.append(np.zeros((25, 3)))

                # process frame
                imgToProcessf = cv2.imread(name)
                imgToProcess = np.fliplr(imgToProcessf)  # flip image to augment dataset
                datum.cvInputData = imgToProcess.copy()
                opWrapper.emplaceAndPop([datum])
                # os.remove(name)

                # Display Image
                # print("Body keypoints: \n" + str(datum.poseKeypoints))
                img = datum.cvOutputData
                # cv2.imshow("frame" + str(currentframe) + " flipped", img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                if datum.poseKeypoints.size > 1:
                    keypoints_flipped.append(datum.poseKeypoints[0][:, :2])  # list append is faster than numpy append
                    keypoints_withConfidence_flipped.append(datum.poseKeypoints[0])
                else:
                    keypoints_flipped.append(np.zeros((25, 2)))
                    keypoints_withConfidence_flipped.append(np.zeros((25, 3)))

        else:
            if currentframe == -1:
                print("Video not found")
                return
            else:

                if not os.path.exists('./Dataset/Keypoints/' + folder):
                    os.makedirs('./Dataset/Keypoints/' + folder)

                keypoints = np.asarray(keypoints)  # numpy arrays are more handy and memory efficient
                keypoints_flipped = np.asarray(keypoints_flipped)  # numpy arrays are more handy and memory efficient
                keypoints_withConfidence = np.asarray(keypoints_withConfidence)
                keypoints_withConfidence_flipped = np.asarray(keypoints_withConfidence_flipped)

                np.save('./Dataset/Keypoints/' + folder + videoname + '.npy', keypoints)
                np.save('./Dataset/Keypoints/' + folder + 'flipped_' + videoname + '.npy', keypoints_flipped)

                np.save('./Dataset/Keypoints/' + folder + videoname + '_with_confidence.npy',
                        keypoints_withConfidence)
                np.save('./Dataset/Keypoints/' + folder + 'flipped_' + videoname + '_with_confidence.npy',
                        keypoints_withConfidence_flipped)
                print("\nVideo ended\n")
                # Release all space and windows once done
                cam.release()
                cv2.destroyAllWindows()
                return True
