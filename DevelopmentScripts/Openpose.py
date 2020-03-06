import sys
import os
import getpass

"""Openpose initialization, see official documentation for further explanations 
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/doc"""

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Change these variables to point to the correct folder (Release/x64 etc.)
    # If you run `make install` (default path is `/usr/local/python` for Ubuntu)
    sys.path.append('/usr/local/python')
    from openpose import pyopenpose as op
except ImportError as e:
    print(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and '
        'have this Python script in the right folder?')
    raise e

username = getpass.getuser()
params = dict()
# params["net_resolution"] = "160x80"
params["model_folder"] = "/home/" + username + "/openpose/models/"


def opinit():
    try:
        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        return op, opWrapper
    except Exception as e:
        print("Openpose init Error")
        sys.exit(-1)
