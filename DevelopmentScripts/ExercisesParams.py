import pickle as pkl

"""
Boolean weights
"""


def paramsSideStepsJacks():
    # Body keypoints {list index : [body component, weight]}
    keyPointsWeights = {
        0: ["Nose", 0],
        1: ["Neck", 0],
        2: ["RShoulder", 1],
        3: ["RElbow", 1],
        4: ["RWrist", 1],
        5: ["LShoulder", 1],
        6: ["LElbow", 1],
        7: ["LWrist", 1],
        8: ["MidHip", 0],
        9: ["RHip", 1],
        10: ["RKnee", 1],
        11: ["RAnkle", 0],
        12: ["LHip", 1],
        13: ["LKnee", 0],
        14: ["LAnkle", 0],
        15: ["REye", 0],
        16: ["LEye", 0],
        17: ["REar", 0],
        18: ["LEar", 0],
        19: ["LBigToe", 0],
        20: ["LSmallToe", 0],
        21: ["LHeel", 0],
        22: ["RBigToe", 0],
        23: ["RSmallToe", 0],
        24: ["RHeel", 0],
        25: ["Background", 0],
    }

    # params
    videoParams = {'slidingWindowDimension': 20,
                   'slidingWindowDimensionUser': 20,
                   'meanRange': 50,
                   'error': 3,
                   }
    params = [keyPointsWeights, videoParams]
    with open('paramsPickle/sidestepsjacks', 'wb') as outFile:
        pkl.dump(params, outFile)


def paramsSquats():
    # Body keypoints {list index : [body component, weight]}
    keyPointsWeights = {
        0: ["Nose", 0],
        1: ["Neck", 0],
        2: ["RShoulder", 0],
        3: ["RElbow", 0],
        4: ["RWrist", 0],
        5: ["LShoulder", 0],
        6: ["LElbow", 0],
        7: ["LWrist", 0],
        8: ["MidHip", 1],
        9: ["RHip", 1],
        10: ["RKnee", 1],
        11: ["RAnkle", 1],
        12: ["LHip", 1],
        13: ["LKnee", 1],
        14: ["LAnkle", 1],
        15: ["REye", 0],
        16: ["LEye", 0],
        17: ["REar", 0],
        18: ["LEar", 0],
        19: ["LBigToe", 0],
        20: ["LSmallToe", 1],
        21: ["LHeel", 1],
        22: ["RBigToe", 0],
        23: ["RSmallToe", 1],
        24: ["RHeel", 1],
        25: ["Background", 0],
    }

    # params
    videoParams = {'slidingWindowDimension': 50,
                   'slidingWindowDimensionUser': 50,
                   'meanRange': 50,
                   'error': 2,
                   }
    params = [keyPointsWeights, videoParams]

    with open('paramsPickle/squats', 'wb') as outFile:
        pkl.dump(params, outFile)


def paramsArmClap():
    # Body keypoints {list index : [body component, weight]}
    keyPointsWeights = {
        0: ["Nose", 0],
        1: ["Neck", 1],
        2: ["RShoulder", 1],
        3: ["RElbow", 1],
        4: ["RWrist", 1],
        5: ["LShoulder", 1],
        6: ["LElbow", 1],
        7: ["LWrist", 1],
        8: ["MidHip", 0],
        9: ["RHip", 0],
        10: ["RKnee", 0],
        11: ["RAnkle", 0],
        12: ["LHip", 0],
        13: ["LKnee", 0],
        14: ["LAnkle", 0],
        15: ["REye", 0],
        16: ["LEye", 0],
        17: ["REar", 0],
        18: ["LEar", 0],
        19: ["LBigToe", 0],
        20: ["LSmallToe", 0],
        21: ["LHeel", 0],
        22: ["RBigToe", 0],
        23: ["RSmallToe", 0],
        24: ["RHeel", 0],
        25: ["Background", 0],
    }

    # params
    videoParams = {'slidingWindowDimension': 30,
                   'slidingWindowDimensionUser': 30,
                   'meanRange': 50,
                   'error': 3,
                   }
    params = [keyPointsWeights, videoParams]
    with open('paramsPickle/armclap', 'wb') as outFile:
        pkl.dump(params, outFile)


def paramsDumbBellCurl():
    # Body keypoints {list index : [body component, weight]}
    keyPointsWeights = {
        0: ["Nose", 0],
        1: ["Neck", 1],
        2: ["RShoulder", 1],
        3: ["RElbow", 1],
        4: ["RWrist", 1],
        5: ["LShoulder", 1],
        6: ["LElbow", 1],
        7: ["LWrist", 1],
        8: ["MidHip", 0],
        9: ["RHip", 0],
        10: ["RKnee", 0],
        11: ["RAnkle", 0],
        12: ["LHip", 0],
        13: ["LKnee", 0],
        14: ["LAnkle", 0],
        15: ["REye", 0],
        16: ["LEye", 0],
        17: ["REar", 0],
        18: ["LEar", 0],
        19: ["LBigToe", 0],
        20: ["LSmallToe", 0],
        21: ["LHeel", 0],
        22: ["RBigToe", 0],
        23: ["RSmallToe", 0],
        24: ["RHeel", 0],
        25: ["Background", 0],
    }

    # params
    videoParams = {'slidingWindowDimension': 50,
                   'slidingWindowDimensionUser': 50,
                   'meanRange': 50,
                   'error': 3,
                   }
    params = [keyPointsWeights, videoParams]
    with open('paramsPickle/dumbbellcurl', 'wb') as outFile:
        pkl.dump(params, outFile)


def paramsPushUps():
    # Body keypoints {list index : [body component, weight]}
    keyPointsWeights = {
        0: ["Nose", 0],
        1: ["Neck", 1],
        2: ["RShoulder", 1],
        3: ["RElbow", 1],
        4: ["RWrist", 1],
        5: ["LShoulder", 1],
        6: ["LElbow", 1],
        7: ["LWrist", 1],
        8: ["MidHip", 1],
        9: ["RHip", 1],
        10: ["RKnee", 0],
        11: ["RAnkle", 0],
        12: ["LHip", 1],
        13: ["LKnee", 0],
        14: ["LAnkle", 0],
        15: ["REye", 0],
        16: ["LEye", 0],
        17: ["REar", 0],
        18: ["LEar", 0],
        19: ["LBigToe", 0],
        20: ["LSmallToe", 0],
        21: ["LHeel", 0],
        22: ["RBigToe", 0],
        23: ["RSmallToe", 0],
        24: ["RHeel", 0],
        25: ["Background", 0],
    }

    # params
    videoParams = {'slidingWindowDimension': 50,
                   'slidingWindowDimensionUser': 50,
                   'meanRange': 50,
                   'error': 3,
                   }
    params = [keyPointsWeights, videoParams]
    with open('paramsPickle/pushups', 'wb') as outFile:
        pkl.dump(params, outFile)


def paramsLunges():
    # Body keypoints {list index : [body component, weight]}
    keyPointsWeights = {
        0: ["Nose", 0],
        1: ["Neck", 0],
        2: ["RShoulder", 0],
        3: ["RElbow", 0],
        4: ["RWrist", 0],
        5: ["LShoulder", 0],
        6: ["LElbow", 0],
        7: ["LWrist", 0],
        8: ["MidHip", 1],
        9: ["RHip", 1],
        10: ["RKnee", 1],
        11: ["RAnkle", 1],
        12: ["LHip", 1],
        13: ["LKnee", 1],
        14: ["LAnkle", 1],
        15: ["REye", 0],
        16: ["LEye", 0],
        17: ["REar", 0],
        18: ["LEar", 0],
        19: ["LBigToe", 1],
        20: ["LSmallToe", 1],
        21: ["LHeel", 1],
        22: ["RBigToe", 1],
        23: ["RSmallToe", 1],
        24: ["RHeel", 1],
        25: ["Background", 0],
    }

    # params
    videoParams = {'slidingWindowDimension': 50,
                   'slidingWindowDimensionUser': 50,
                   'meanRange': 50,
                   'error': 3,
                   }
    params = [keyPointsWeights, videoParams]
    with open('paramsPickle/lunges', 'wb') as outFile:
        pkl.dump(params, outFile)


def getPickle():
    paramsSideStepsJacks()
    paramsSquats()
    paramsArmClap()
    paramsDumbBellCurl()
    paramsPushUps()
    paramsLunges()
