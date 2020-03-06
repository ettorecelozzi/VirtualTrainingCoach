import pickle as pkl


# 25/10 = 0,4
# sum weight = 10
def paramsSideStepsJacks():
    # Body keypoints {list index : [body component, weight]}
    keyPointsWeights = {
        0: ["Nose", 0.0],
        1: ["Neck", 0.0],
        2: ["RShoulder", 0.6],
        3: ["RElbow", 0.7],
        4: ["RWrist", 0.7],
        5: ["LShoulder", 0.6],
        6: ["LElbow", 0.7],
        7: ["LWrist", 0.7],
        8: ["MidHip", 0.0],
        9: ["RHip", 0.2],
        10: ["RKnee", 0.7],
        11: ["RAnkle", 0.0],
        12: ["LHip", 0.2],
        13: ["LKnee", 0.7],
        14: ["LAnkle", 0.0],
        15: ["REye", 0.0],
        16: ["LEye", 0.0],
        17: ["REar", 0.0],
        18: ["LEar", 0.0],
        19: ["LBigToe", 0.0],
        20: ["LSmallToe", 0.0],
        21: ["LHeel", 0.0],
        22: ["RBigToe", 0.0],
        23: ["RSmallToe", 0.0],
        24: ["RHeel", 0.0],
        25: ["Background", 0.0],
    }

    # params
    videoParams = {'slidingWindowDimension': 20,
                   'slidingWindowDimensionUser': 20,
                   'meanRange': 50,
                   'firstMin_TrainerCycle': 0,
                   'error': 3,
                   }
    params = [keyPointsWeights, videoParams]
    with open('paramsPickle/SideStepsJacks', 'wb') as outFile:
        pkl.dump(params, outFile)


def paramsPopSquats():
    # Body keypoints {list index : [body component, weight]}
    keyPointsWeights = {
        0: ["Nose", 0.0],
        1: ["Neck", 0.0],
        2: ["RShoulder", 0.6],
        3: ["RElbow", 0.0],
        4: ["RWrist", 0.0],
        5: ["LShoulder", 0.6],
        6: ["LElbow", 0.0],
        7: ["LWrist", 0.0],
        8: ["MidHip", 0.3],
        9: ["RHip", 1.5],
        10: ["RKnee", 1.9],
        11: ["RAnkle", 0.7],
        12: ["LHip", 1.5],
        13: ["LKnee", 1.9],
        14: ["LAnkle", 0.7],
        15: ["REye", 0.0],
        16: ["LEye", 0.0],
        17: ["REar", 0.0],
        18: ["LEar", 0.0],
        19: ["LBigToe", 0.0],
        20: ["LSmallToe", 0.0],
        21: ["LHeel", 0.0],
        22: ["RBigToe", 0.0],
        23: ["RSmallToe", 0.0],
        24: ["RHeel", 0.0],
        25: ["Background", 0.0],
    }

    # params
    videoParams = {'slidingWindowDimension': 20,
                   'slidingWindowDimensionUser': 30,
                   'meanRange': 50,
                   'firstMin_TrainerCycle': 0,
                   'error': 2,
                   }
    params = [keyPointsWeights, videoParams]

    with open('paramsPickle/PopSquats', 'wb') as outFile:
        pkl.dump(params, outFile)


def paramsLegLift():
    # Body keypoints {list index : [body component, weight]}
    keyPointsWeights = {
        0: ["Nose", 0.0],
        1: ["Neck", 0.0],
        2: ["RShoulder", 0.0],
        3: ["RElbow", 0.0],
        4: ["RWrist", 0.0],
        5: ["LShoulder", 0.0],
        6: ["LElbow", 0.0],
        7: ["LWrist", 0.0],
        8: ["MidHip", 0.8],
        9: ["RHip", 1.0],
        10: ["RKnee", 2.0],
        11: ["RAnkle", 1.5],
        12: ["LHip", 1.0],
        13: ["LKnee", 2.0],
        14: ["LAnkle", 1.5],
        15: ["REye", 0.0],
        16: ["LEye", 0.0],
        17: ["REar", 0.0],
        18: ["LEar", 0.0],
        19: ["LBigToe", 0.0],
        20: ["LSmallToe", 0.8],
        21: ["LHeel", 0.0],
        22: ["RBigToe", 0.0],
        23: ["RSmallToe", 0.8],
        24: ["RHeel", 0.0],
        25: ["Background", 0.0],
    }

    # params
    videoParams = {'slidingWindowDimension': 20,
                   'slidingWindowDimensionUser': 30,
                   'meanRange': 50,
                   'firstMin_TrainerCycle': 0,
                   'error': 2,
                   }
    params = [keyPointsWeights, videoParams]

    with open('paramsPickle/LegLift', 'wb') as outFile:
        pkl.dump(params, outFile)


def paramsWarriorJacks():
    # Body keypoints {list index : [body component, weight]}
    keyPointsWeights = {
        0: ["Nose", 0.0],
        1: ["Neck", 0.0],
        2: ["RShoulder", 0.6],
        3: ["RElbow", 0.7],
        4: ["RWrist", 0.7],
        5: ["LShoulder", 0.6],
        6: ["LElbow", 0.7],
        7: ["LWrist", 0.7],
        8: ["MidHip", 0.0],
        9: ["RHip", 0.2],
        10: ["RKnee", 0.7],
        11: ["RAnkle", 0.0],
        12: ["LHip", 0.2],
        13: ["LKnee", 0.7],
        14: ["LAnkle", 0.0],
        15: ["REye", 0.0],
        16: ["LEye", 0.0],
        17: ["REar", 0.0],
        18: ["LEar", 0.0],
        19: ["LBigToe", 0.0],
        20: ["LSmallToe", 0.0],
        21: ["LHeel", 0.0],
        22: ["RBigToe", 0.0],
        23: ["RSmallToe", 0.0],
        24: ["RHeel", 0.0],
        25: ["Background", 0.0],
    }

    # params
    videoParams = {'slidingWindowDimension': 120,
                   'slidingWindowDimensionUser': 120,
                   'meanRange': 50,
                   'firstMin_TrainerCycle': 0,
                   'error': 3,
                   }
    params = [keyPointsWeights, videoParams]
    with open('paramsPickle/WarriorJacks', 'wb') as outFile:
        pkl.dump(params, outFile)


def paramsArmsLift():
    # Body keypoints {list index : [body component, weight]}
    keyPointsWeights = {
        0: ["Nose", 0.0],
        1: ["Neck", 0.0],
        2: ["RShoulder", 1.0],
        3: ["RElbow", 3.5],
        4: ["RWrist", 0.5],
        5: ["LShoulder", 1.0],
        6: ["LElbow", 3.5],
        7: ["LWrist", 0.5],
        8: ["MidHip", 0.0],
        9: ["RHip", 0.0],
        10: ["RKnee", 0.0],
        11: ["RAnkle", 0.0],
        12: ["LHip", 0.0],
        13: ["LKnee", 0.0],
        14: ["LAnkle", 0.0],
        15: ["REye", 0.0],
        16: ["LEye", 0.0],
        17: ["REar", 0.0],
        18: ["LEar", 0.0],
        19: ["LBigToe", 0.0],
        20: ["LSmallToe", 0.0],
        21: ["LHeel", 0.0],
        22: ["RBigToe", 0.0],
        23: ["RSmallToe", 0.0],
        24: ["RHeel", 0.0],
        25: ["Background", 0.0],
    }

    # params
    videoParams = {'slidingWindowDimension': 70,
                   'slidingWindowDimensionUser': 70,
                   'meanRange': 50,
                   'firstMin_TrainerCycle': 0,
                   'error': 3,
                   }
    params = [keyPointsWeights, videoParams]
    with open('paramsPickle/ArmsLift', 'wb') as outFile:
        pkl.dump(params, outFile)


def paramsOneLegLift():
    # Body keypoints {list index : [body component, weight]}
    keyPointsWeights = {
        0: ["Nose", 0.0],
        1: ["Neck", 0.0],
        2: ["RShoulder", 0.0],
        3: ["RElbow", 0.0],
        4: ["RWrist", 0.0],
        5: ["LShoulder", 0.0],
        6: ["LElbow", 0.0],
        7: ["LWrist", 0.0],
        8: ["MidHip", 0.0],
        9: ["RHip", 0.5],
        10: ["RKnee", 1.0],
        11: ["RAnkle", 3.5],
        12: ["LHip", 0.5],
        13: ["LKnee", 1.0],
        14: ["LAnkle", 3.5],
        15: ["REye", 0.0],
        16: ["LEye", 0.0],
        17: ["REar", 0.0],
        18: ["LEar", 0.0],
        19: ["LBigToe", 0.0],
        20: ["LSmallToe", 0.0],
        21: ["LHeel", 0.0],
        22: ["RBigToe", 0.0],
        23: ["RSmallToe", 0.0],
        24: ["RHeel", 0.0],
        25: ["Background", 0.0],
    }

    # params
    videoParams = {'slidingWindowDimension': 5,
                   'slidingWindowDimensionUser': 5,
                   'meanRange': 50,
                   'firstMin_TrainerCycle': 0,
                   'error': 3,
                   }
    params = [keyPointsWeights, videoParams]
    with open('paramsPickle/OneLegLift', 'wb') as outFile:
        pkl.dump(params, outFile)


def getPickle():
    paramsOneLegLift()
    paramsSideStepsJacks()
    paramsPopSquats()
    paramsLegLift()
    paramsWarriorJacks()
    paramsArmsLift()
