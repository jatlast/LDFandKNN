########################################################################
# Jason Baumbach
#   CSC 546 - Project (due: April 06, 2019 @ 5:59 PM)
#
# Note: this code is available on GitHub 
#   https://github.com/jatlast/kNearestNeighbor.git
#
########################################################################

# required for reading csv files to get just the header
import csv
# required for sqrt function in Euclidean Distance calculation
import math
# required for parsing data files
import re

# allow command line options
import argparse
parser = argparse.ArgumentParser(description="Perform k-nearest neighbor classification on 2-dimensional data provided. (Note: 2D files include 3 columns: x, y, class)")
parser.add_argument("-k", "--kneighbors", type=int, choices=range(1, 30), default=3, help="number of nearest neighbors to use")
parser.add_argument("-ft", "--filetrain", default="../uci_hd_preprocessing/data/cleveland_smoke_uci+_normal_train.csv", help="training file name (and path if not in . dir)")
parser.add_argument("-fs", "--filetest", default="../uci_hd_preprocessing/data/cleveland_smoke_uci+_normal_test.csv", help="testing file name (and path if not in . dir)")
parser.add_argument("-tn", "--targetname", default="target", help="the name of the target attribute")
#parser.add_argument("-t2", "--targetsecondary", default="target", help="the name of the secondary target attribute")
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2, 3], default=0, help="increase output verbosity")
args = parser.parse_args()

# create a dictionary of list objects equal to the number of neighbors to test
neighbors_dict = {}
for i in range(1, args.kneighbors + 1):
    neighbors_dict[i] = {'index' : -1, 'distance' : 1000, 'type' : ''}

if args.verbosity > 0:
    print(f"neighbors={args.kneighbors} : len(neighbors_dict)={len(neighbors_dict)}")
#    print(f"filename={args.filename} : neighbors={args.kneighbors} : len(neighbors_dict)={len(neighbors_dict)}")

# compute Euclidean distance between any two given vectors with any length.
# Note: adapted from CSC 587 Adv Data Mining, HW02
# Note: a return value < 0 = Error
def EuclideanDistanceBetweenTwoVectors(vOne, vTwo):
    distance = 0
    v_one_len = len(vOne)
    v_two_len = len(vTwo)
    # vOne & vTwo must be of equal length
    if(v_one_len != v_two_len):
        print(f"Warning UD: {v_one_len} != {v_two_len}")
        return -1

    for p in range(0, v_one_len):
        distance += math.pow((abs(float(vOne[p]) - float(vTwo[p]))), 2)
    return math.sqrt(distance)

# get the inner product (dot product) of two equal length vectors
def GetInnerProductOfTwoVectors(vOne, vTwo):
    product = 0
    v_one_len = len(vOne)
    v_two_len = len(vTwo)
    # vOne & vTwo must be of equal length
    if(v_one_len != v_two_len):
        print(f"Warning DP: {v_one_len} != {v_two_len}")
        return -1
    else:
        for i in range(0, v_one_len):
            product += float(vOne[i]) * float(vTwo[i])
    return product

# variables that are useful to pass around
variables_dict = {
    'training_file' : args.filetrain
    , 'testing_file' : args.filetest
    , 'target_col_name' : args.targetname
    , 'target_col_index_train' : -1
    , 'target_col_index_test' : -1
    # , 'target2_col_name' : args.targetsecondary
    # , 'target2_col_index_train' : -1
    # , 'target2_col_index_test' : -1
    , 'training_col_count' : -1
    , 'testing_col_count' : -1
    , 'plot_title' : 'Default Title'
    , 'knn_majority_type' : 0
    , 'test_runs_count' : 0
    , 'com_knn_ldf_right' : 0
    , 'com_knn_right_ldf_wrong' : 0
    , 'com_ldf_right_knn_wrong' : 0
    , 'com_ldf_wrong_knn_right' : 0
    , 'com_knn_wrong_ldf_right' : 0
    , 'com_knn_ldf_wrong' : 0
    , 'ldf_confidence_zero' : -1
    , 'ldf_diff_min' : 1000
    , 'ldf_diff_max' : 0
    , 'classification' : 'UNK'
    , 'ignore_columns' : ['cigs', 'years']
    , 'knn_confusion_matrix' : {
        'TN'    : 0
        , 'FP'  : 0
        , 'FN'  : 0
        , 'TP'  : 0
    }
    , 'ldf_confusion_matrix' : {
        'TN'    : 0
        , 'FP'  : 0
        , 'FN'  : 0
        , 'TP'  : 0
    }
    , 'com_confusion_matrix' : {
        'TN'    : 0
        , 'FP'  : 0
        , 'FN'  : 0
        , 'TP'  : 0
    }
}

# specific to UCI's Heart Disease data set which has two target columns: num (0-4) & target (0 or 1)
# This allows the code to ignore the unmentioned column dynamically
if args.targetname == 'num':
    variables_dict['ignore_columns'].append('target')
elif args.targetname == 'target':
    variables_dict['ignore_columns'].append('num')

# Read the provided data files
# Note: These files are not in a generic (',' or '\t') delimited format -- they require parsing
def ReadFileDataIntoDictOfLists(sFileName, dDictOfLists):
    # read the file
    with open(sFileName) as csv_file:
        line_number = 0
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            dDictOfLists[line_number] = row
            line_number += 1

# Get the class target(s) index(es) for dynamic inclusion/exclusion later
def AddTargetIndexesToVarDict(dDictOfLists, dVariables):
    col_count = len(dDictOfLists[0])
    # save the column counts to the variables dictionary
    if dDictOfLists['type'] == 'training':
        dVariables['training_col_count'] = col_count
    elif dDictOfLists['type'] == 'testing':
        dVariables['testing_col_count'] = col_count
    else:
        print(f"Warning: type {dDictOfLists['type']} not recognized.")
    # loop through the header to find and save the target column index(es) to the variables dictionary
    for col in range(0, col_count):
        if dDictOfLists[0][col] == dVariables['target_col_name']:
            if dDictOfLists['type'] == 'training':
                dVariables['target_col_index_train'] = col
            elif dDictOfLists['type'] == 'testing':
                dVariables['target_col_index_test'] = col
        # elif dDictOfLists[0][col] == dVariables['target2_col_name']:
        #     if dDictOfLists['type'] == 'training':
        #         dVariables['target2_col_index_train'] = col
        #     elif dDictOfLists['type'] == 'testing':
        #         dVariables['target2_col_index_test'] = col

def AddTargetTypesToVarDict(dTrainingData, dVariables):
    dVariables['target_types'] = {}
    # dVariables['target2_types'] = {}
    for i in range(1, len(dTrainingData) - 1):
        if dTrainingData[i][dVariables['target_col_index_train']] not in dVariables['target_types']:
            dVariables['target_types'][dTrainingData[i][dVariables['target_col_index_train']]] = dTrainingData[i][dVariables['target_col_index_train']]
            #print(f"row:{i} target:{dVariables['target_types'][dTrainingData[i][dVariables['target_col_index_train']]]}")
        # elif dTrainingData[i][dVariables['target2_col_index_train']] not in dVariables['target2_types']:
        #     dVariables['target_types'][dTrainingData[i][dVariables['target2_col_index_train']]] = dTrainingData[i][dVariables['target2_col_index_train']]

def AddSharedAttributesToVarDict(dTrainingData, dTestingData, dVariables):
    dVariables['shared_attributes'] = []
    # check which data set is larger then loop through the smaller on the outside
    if dVariables['training_col_count'] < dVariables['testing_col_count']:
        # print a warning if there are more testing than trainging attributes
        print(f"Warning: There are more testing attributes {dVariables['testing_col_count']} than training attributes {dVariables['training_col_count']}.")
        for i in dTrainingData[0]:
            if i not in dVariables['ignore_columns']:
                for j in dTestingData[0]:
                    if i == j:
                        dVariables['shared_attributes'].append(i)
    # the testing data set is smaller so loop through it on the outside
    else:
        for i in dTestingData[0]:
            if i not in dVariables['ignore_columns']:
                for j in dTrainingData[0]:
                    if i == j:
                        dVariables['shared_attributes'].append(i)
        
# create a vector of only the shared attributes (excluding target attributes)
def ReturnVectorOfOnlySharedAttributes(dDictOfLists, sIndex, dVariables):
    header_vec = [] # for debugging only
    return_vec = []
    col_count = 0
    # set the appropriate col_count by data set type
    if dDictOfLists['type'] == 'training':
        col_count = dVariables['training_col_count']
    elif dDictOfLists['type'] == 'testing':
        col_count = dVariables['testing_col_count']
    else:
        print(f"Warning: type {dDictOfLists['type']} not recognized.")
    # loop through the header to find and save the target column index(es) to the variables dictionary
    for i in range(0, col_count):
        # ignore the target attributes
        # if i == dVariables['target_col_index_test'] or i == dVariables['target2_col_index_test']:
        if i == dVariables['target_col_index_train'] and dDictOfLists['type'] == 'training':
            continue
        elif i == dVariables['target_col_index_test'] and dDictOfLists['type'] == 'testing':
            continue
        # append only the attributes shared with the training data
        for col in dVariables['shared_attributes']:
            if dDictOfLists[0][i] == col:
                return_vec.append(dDictOfLists[sIndex][i])
                header_vec.append(col)

    # debugging info
    if args.verbosity > 2:
        print(f"shared {dDictOfLists['type']}:{header_vec}")
    
    return return_vec

# Populate the k-nearest neighbors by comparing all training data with test data point
def PopulateNearestNeighborsDicOfIndexes(dNeighbors, dTrainingData, vTestData, dVariables):
    distances = []  # for debugging only (store then sort all distances for comparison to the chosen distances)
    # Loop through the training set to find the least distance(s)
    for i in range(1, len(dTrainingData) - 1):
        train_vec = ReturnVectorOfOnlySharedAttributes(dTrainingData, i, dVariables)
        EuclideanDistance = EuclideanDistanceBetweenTwoVectors(vTestData, train_vec)
        distances.append(EuclideanDistance) # for debugging only
        neighbor_max_index = -1
        neighbor_max_value = -1
        # Loop through the neighbors dict so the maximum stored is always replaced first
        for j in range(1, len(dNeighbors) + 1):
            if dNeighbors[j]['distance'] > neighbor_max_value:
                neighbor_max_value = dNeighbors[j]['distance']
                neighbor_max_index = j
        # save the newest least distance over the greatest existing neighbor distance
        if EuclideanDistance < neighbor_max_value:
            dNeighbors[neighbor_max_index]['distance'] = EuclideanDistance
            dNeighbors[neighbor_max_index]['index'] = i
#            dNeighbors[neighbor_max_index]['type'] = int(float(dTrainingData[i][dVariables['target_col_index_train']]))
            dNeighbors[neighbor_max_index]['type'] = dTrainingData[i][dVariables['target_col_index_train']]

    # debugging: print the least distances out of all distances calculated
    if args.verbosity > 2:
        distances.sort()
        print("least distances:")
        for i in range(0, len(dNeighbors)):
            print(f"min{i}:({distances[i]}) \t& neighbors:({dNeighbors[i+1]['distance']})")

# Return the "type" value of the majority of nearest neighbors
def AddKNNMajorityTypeToVarDict(dNeighbors, dVariables):
    type_count_dict = {}
    dVariables['knn_majority_type'] = -1
    dVariables['knn_majority_count'] = 0
    dVariables['knn_secondary_type'] = -1
    dVariables['knn_secondary_count'] = 0
    dVariables['knn_confidence'] = -1
    dVariables['classification'] = 'UNK'
    dVariables['knn_majority_count'] = 0
    # loop through the target types and zero out the type_count_dict
    for key in dVariables['target_types']:
        type_count_dict[key] = 0
#        print(f"key:{key}:{type_count_dict[key]}")

    # loop through the nearest neighbors and total the different type hits
    for i in range(1, len(dNeighbors) + 1):
        type_count_dict[dNeighbors[i]['type']] += 1
#        print(f"key2:{i}:{dNeighbors[i]['type']}:{type_count_dict[dNeighbors[i]['type']]}")

    # loop through the type_count_dict to find the largest type count value
    for key in type_count_dict:
#        print(f"key3:{key}:{type_count_dict[key]}")

        # store the largest and second largest g(x) for later comparison to determine confidence
        # current better than second best
        if dVariables['knn_secondary_count'] < type_count_dict[key]:
            # current better than best
            if dVariables['knn_majority_count'] < type_count_dict[key]:
                # set best to current
                dVariables['knn_majority_count'] = type_count_dict[key]
                dVariables['knn_majority_type'] = key
            else:
                # set second best to current best
                dVariables['knn_secondary_count'] = type_count_dict[key]
                dVariables['ldf_second_best_target'] = key
        # current better than best
        elif dVariables['knn_majority_count'] < type_count_dict[key]:
            # set second best to previous best
            dVariables['knn_secondary_count'] = dVariables['knn_majority_count']
            dVariables['ldf_second_best_target'] = dVariables['knn_majority_type']
            # set best to current
            dVariables['knn_majority_count'] = type_count_dict[key]
            dVariables['knn_majority_type'] = key

    # only calculate confidence if majority != secondary
    if dVariables['knn_majority_count'] != dVariables['knn_secondary_count']:
        dVariables['knn_confidence'] = dVariables['knn_majority_count'] / (dVariables['knn_majority_count'] + dVariables['knn_secondary_count'])
    else:
        dVariables['knn_confidence'] = 1
    
    if args.verbosity > 2:
        print(f"majority:{dVariables['knn_majority_type']}{type_count_dict}")
        # for key, value in type_count_dict.items():
        #     print(f"{key}:{value}")

# calculate the class type means from the training data
# For PluginLDF functionality
def AddTargetTypeMeansToVarDict(dTrainingData, dVariables):
    col_sums_dic = {}
    row_count_dic = {}
#    dVariables['target_type_means'] = {}

    # initialize the col_sums_dic to zeros
    for key in dVariables['target_types']:
        col_sums_dic[key] = {}
        row_count_dic[key] = 0
        dVariables[key] = {'ldf_mean' : []}
        for col in dVariables['shared_attributes']:
            if col != dVariables['target_col_name']:
                col_sums_dic[key][col] = 0

    # Loop through the training set to calculate the totals required to calculate the means
    # loop through the traing set rows
    for i in range(1, len(dTrainingData) - 1):
        # loop through the traing set columns
        for j in range(0, len(dTrainingData[0])):
            # loop through the shared columns
            for col in dVariables['shared_attributes']:
                # check if the column is shared
                if dTrainingData[0][j] == col:
                    if col != dVariables['target_col_name']:
                        # sum the colum values
                        col_sums_dic[dTrainingData[i][dVariables['target_col_index_train']]][col] += float(dTrainingData[i][j])
                    else:
                        # incrament the column count
                        row_count_dic[dTrainingData[i][dVariables['target_col_index_train']]] += 1

    # calculate the means
    for key in dVariables['target_types']:
        for col in col_sums_dic[key]:
            # debug info
            if args.verbosity > 2:
                print(f"col:{col}:\t{col_sums_dic[key][col]} / {row_count_dic[key]}")
            # append the colum mean to the target type mean vector
            if row_count_dic[key] > 0:
                dVariables[key]['ldf_mean'].append(col_sums_dic[key][col] / row_count_dic[key])
            else:
                print(f"Warning: LDF mean = 0 for target:{key}")
                print(f"col:{col}:\t{col_sums_dic[key][col]} / {row_count_dic[key]}")
                dVariables[key]['ldf_mean'].append(0)

# get the largest PluginLDF value of g(x)
def AddCalculatesOfPluginLDFToVarDic(vTestData, dVariables):
    dVariables['ldf_best_g'] = -1
    dVariables['ldf_second_best_g'] = -1
    dVariables['ldf_best_target'] = -1
    dVariables['ldf_second_best_target'] = -1
    dVariables['ldf_confidence'] = -1
    ldf_diff = -1
    # loop through the target types
    for key in dVariables['target_types']:
        # calculate the inner (dot) products of the target type means
        dVariables[key]['ldf_dot_mean'] = GetInnerProductOfTwoVectors(vTestData, dVariables[key]['ldf_mean'])
        # calculate g(x)
        dVariables[key]['ldf_g'] = (2 * dVariables[key]['ldf_dot_mean']) - dVariables[key]['ldf_mean_square']

        # store the largest and second largest g(x) for later comparison to determine confidence
        # current better than second best
        if dVariables['ldf_second_best_g'] < dVariables[key]['ldf_g']:
            # current better than best
            if dVariables['ldf_best_g'] < dVariables[key]['ldf_g']:
                # set best to current
                dVariables['ldf_best_g'] = dVariables[key]['ldf_g']
                dVariables['ldf_best_target'] = key
            else:
                # set second best to current best
                dVariables['ldf_second_best_g'] = dVariables[key]['ldf_g']
                dVariables['ldf_second_best_target'] = key
        # current better than best
        elif dVariables['ldf_best_g'] < dVariables[key]['ldf_g']:
            # set second best to previous best
            dVariables['ldf_second_best_g'] = dVariables['ldf_best_g']
            dVariables['ldf_second_best_target'] = dVariables['ldf_best_target']
            # set best to current
            dVariables['ldf_best_g'] = dVariables[key]['ldf_g']
            dVariables['ldf_best_target'] = key

        # debug info: print the formul used
        if args.verbosity > 2:
            print(f"\t{key}_g(x): {round(dVariables[key]['ldf_g'], 2)} = (2 * {round(dVariables[key]['ldf_dot_mean'], 2)}) - {round(dVariables[key]['ldf_mean_square'], 2)}")

    # get the difference between best and second best
    ldf_diff = dVariables['ldf_best_g'] - dVariables['ldf_second_best_g']
    # set the max
    if dVariables['ldf_diff_max'] < ldf_diff:
        dVariables['ldf_diff_max'] = ldf_diff
    
    # set the min
    if dVariables['ldf_diff_min'] > ldf_diff:
        dVariables['ldf_diff_min'] = ldf_diff
    
    # use min-max to calculate confidence if min & max have been initialized
    if dVariables['ldf_diff_max'] != dVariables['ldf_diff_min']:
        dVariables['ldf_confidence'] = ((ldf_diff - dVariables['ldf_diff_min']) / (dVariables['ldf_diff_max'] - dVariables['ldf_diff_min']))
    else:
        dVariables['ldf_confidence'] = ldf_diff

    # sum all LDF confidenc <= 0
    if dVariables['ldf_confidence'] <= 0:
        dVariables['ldf_confidence_zero'] += 1
        if args.verbosity > 2:
            print(f"ldf diff:{dVariables['ldf_best_g']} - {dVariables['ldf_second_best_g']}")

def TrackConfusionMatrixSums(sTestType, sPredictionType, sPrefix, dVariables):
    # store the confusion matrix counts (unfortunately this makes the code dependant on numerical target values)
    if int(float(sTestType)) > 0:
        if sTestType == sPredictionType:
            dVariables[sPrefix + '_classification'] = 'Correct'
            dVariables[sPrefix + '_confusion_matrix']['TP'] += 1
        else:
            dVariables[sPrefix + '_classification'] = 'Incorrect'
            dVariables[sPrefix + '_confusion_matrix']['FP'] += 1
    else:
        if sTestType == sPredictionType:
            dVariables[sPrefix + '_classification'] = 'Correct'
            dVariables[sPrefix + '_confusion_matrix']['TN'] += 1
        else:
            dVariables[sPrefix + '_classification'] = 'Incorrect'
            dVariables[sPrefix + '_confusion_matrix']['FN'] += 1

def PrintConfusionMatrix(sPrefix, dVariables):
    print(f"\n{sPrefix} - Confusion Matrix:\n\tTP:{dVariables[sPrefix + '_confusion_matrix']['TP']} | FN:{dVariables[sPrefix + '_confusion_matrix']['FN']}\n\tFP:{dVariables[sPrefix + '_confusion_matrix']['FP']} | TN:{dVariables[sPrefix + '_confusion_matrix']['TN']}")
    dVariables[sPrefix + '_accuracy'] = round((dVariables[sPrefix + '_confusion_matrix']['TP'] + dVariables[sPrefix + '_confusion_matrix']['TN']) / (dVariables[sPrefix + '_confusion_matrix']['TP'] + dVariables[sPrefix + '_confusion_matrix']['TN'] + dVariables[sPrefix + '_confusion_matrix']['FP'] + dVariables[sPrefix + '_confusion_matrix']['FN']),2)
    dVariables[sPrefix + '_error_rate'] = round((1 - dVariables[sPrefix + '_accuracy']),2)
    dVariables[sPrefix + '_precision'] = round(dVariables[sPrefix + '_confusion_matrix']['TP'] / (dVariables[sPrefix + '_confusion_matrix']['TP'] + dVariables[sPrefix + '_confusion_matrix']['FP']),2)
    dVariables[sPrefix + '_specificity'] = round(dVariables[sPrefix + '_confusion_matrix']['TN'] / (dVariables[sPrefix + '_confusion_matrix']['TN'] + dVariables[sPrefix + '_confusion_matrix']['FN']),2)
    dVariables[sPrefix + '_FPR'] = round(dVariables[sPrefix + '_confusion_matrix']['FP'] / (dVariables[sPrefix + '_confusion_matrix']['TN'] + dVariables[sPrefix + '_confusion_matrix']['FN']),2)
    print(f"Accuracy   :{dVariables[sPrefix + '_accuracy']}")
    print(f"Error Rate :{dVariables[sPrefix + '_error_rate']}")
    print(f"Precision  :{dVariables[sPrefix + '_precision']}")
    print(f"Specificity:{dVariables[sPrefix + '_specificity']}")
    print(f"FPR        :{dVariables[sPrefix + '_FPR']}")

def AddRunningPredictionStatsToVarDict(sTestType, dVariables):
    dVariables['test_runs_count'] += 1
    # COM-bination right
    if sTestType == dVariables['com_best_target']:
        # KNN right
        if sTestType == dVariables['knn_majority_type']:
            # LDF right
            if sTestType == dVariables['ldf_best_target']:
                dVariables['com_knn_ldf_right'] += 1
            # LDF wrong
            else:
                dVariables['com_knn_right_ldf_wrong'] += 1
        # KNN wrong
        else:
            # LDF right
            if sTestType == dVariables['ldf_best_target']:
                dVariables['com_ldf_right_knn_wrong'] += 1
            # LDF wrong
            else:
                # this should never happen!
                print("Warning: COM-bination can never be connrect when both KNN & LDF are incorrect.")
    # COM-bination wrong
    else:
        # KNN right
        if sTestType == dVariables['knn_majority_type']:
            # LDF right
            if sTestType == dVariables['ldf_best_target']:
                # this should never happen!
                print("Warning: COM-bination can never be inconnrect when both KNN & LDF are correct.")
            # LDF wrong
            else:
                dVariables['com_ldf_wrong_knn_right'] += 1
        # KNN wrong
        else:
            # LDF right
            if sTestType == dVariables['ldf_best_target']:
                dVariables['com_knn_wrong_ldf_right'] += 1
            # LDF wrong
            else:
                dVariables['com_knn_ldf_wrong'] += 1

# Load the training data
training_dict = {'type' : 'training'}
ReadFileDataIntoDictOfLists(variables_dict['training_file'], training_dict)

# add the target indexes of the training set to the variables dictionary
AddTargetIndexesToVarDict(training_dict, variables_dict)

# add the possible target types to the variables dictionary
AddTargetTypesToVarDict(training_dict, variables_dict)
# print debugging info
if args.verbosity > 0:
    for key in variables_dict['target_types'].values():
        print(f"target types {key}:{variables_dict['target_types'][key]}")

# Load the testing data
testing_dict = {'type' : 'testing'}
ReadFileDataIntoDictOfLists(variables_dict['testing_file'], testing_dict)

# get the target indexes of the testing set
AddTargetIndexesToVarDict(testing_dict, variables_dict)

# get the shared attributes for comparing testing data with training data
AddSharedAttributesToVarDict(training_dict, testing_dict, variables_dict)
if args.verbosity > 0:
    print(f"shared attributes:{variables_dict['shared_attributes']}")
    print(f"vector attributes:{ReturnVectorOfOnlySharedAttributes(testing_dict, 0, variables_dict)}")

# For PluginLDF functionality
AddTargetTypeMeansToVarDict(training_dict, variables_dict)

# calculate the inner (dot) products of the two class type means
for key in variables_dict['target_types']:
    variables_dict[key]['ldf_mean_square'] = GetInnerProductOfTwoVectors(variables_dict[key]['ldf_mean'], variables_dict[key]['ldf_mean'])

# print debug info
if args.verbosity > 1:
    for key in variables_dict['target_types']:
        print(f"{key} mean_sq:{variables_dict[key]['ldf_mean_square']} | mean:{variables_dict[key]['ldf_mean']}")

#if args.verbosity > 0:
# print the trainging and testing data lengths (subtract 1 for headers and 1 for starting from zero)
print(f"train:{len(training_dict)-2} | test:{len(testing_dict)-2}")

# Print some of the input file data
if args.verbosity > 1:
    print(f"The first 5 training samples with target:{training_dict[0][variables_dict['target_col_index_train']]}:")
    # if variables_dict['target2_col_index_train'] >= 0:
    #     print(f"\t(& target2:{training_dict[0][variables_dict['target2_col_index_train']]})")
    for i in range(0, 5):
        print(f"\t{i} {training_dict[i]}")
    print(f"\nThe first 5 testing samples with target:{testing_dict[0][variables_dict['target_col_index_test']]}")
    # if variables_dict['target2_col_index_train'] >= 0:
    #     print(f"\t(& target2:{testing_dict[0][variables_dict['target2_col_index_test']]})")
    for i in range(0, 5):
        print(f"\t{i} {testing_dict[i]}")

# loop through all testing data
#if 1 == 0:
for i in range(1, len(testing_dict) - 1):
#for i in range(1, 3):
    test_vec = ReturnVectorOfOnlySharedAttributes(testing_dict, i, variables_dict)

    # get the k-nearest neighbors
    PopulateNearestNeighborsDicOfIndexes(neighbors_dict, training_dict, test_vec, variables_dict)
#    print(f"neighbors:{neighbors_dict}")

    # add the k-nearest neighbors' "knn_majority_type" to the variables_dict
    AddKNNMajorityTypeToVarDict(neighbors_dict, variables_dict)

    # add the calculations fo the PluginLDF values to the variables dict inclulding g(x)    
    AddCalculatesOfPluginLDFToVarDic(test_vec, variables_dict)

    # ----- Store the Confusion Matrix running counts -----
    # store the confusion matrix counts for KNN
    TrackConfusionMatrixSums(testing_dict[i][variables_dict['target_col_index_test']], variables_dict['knn_majority_type'], 'knn', variables_dict)
    # store the confusion matrix counts for KNN
    TrackConfusionMatrixSums(testing_dict[i][variables_dict['target_col_index_test']], variables_dict['ldf_best_target'], 'ldf', variables_dict)
    # store the confusion matrix counts for most confident prediction
    if variables_dict['knn_confidence'] > variables_dict['ldf_confidence']:
        variables_dict['com_best_target'] = variables_dict['knn_majority_type']
        # debugging infoi
        if args.verbosity > 1:
            if variables_dict['ldf_best_target'] != variables_dict['knn_majority_type']:
                print(f"KNN:{variables_dict['knn_majority_type']}>({testing_dict[i][variables_dict['target_col_index_test']]}:confidence) KNN:{variables_dict['knn_majority_type']}:{round(variables_dict['knn_confidence'],2)} | LDF:{variables_dict['ldf_best_target']}:{round(variables_dict['ldf_confidence'],2)}")
    else:
        variables_dict['com_best_target'] = variables_dict['ldf_best_target']
        # debugging infoi
        if args.verbosity > 1:
            if variables_dict['ldf_best_target'] != variables_dict['knn_majority_type']:
                print(f"LDF:{variables_dict['com_best_target']}>({testing_dict[i][variables_dict['target_col_index_test']]}:confidence) KNN:{variables_dict['knn_majority_type']}:{round(variables_dict['knn_confidence'],2)} | LDF:{variables_dict['ldf_best_target']}:{round(variables_dict['ldf_confidence'],2)}")
    TrackConfusionMatrixSums(testing_dict[i][variables_dict['target_col_index_test']], variables_dict['com_best_target'], 'com', variables_dict)
    # -----------------------------------------------------

    # add the running totals of knn & ldf's predictions
    AddRunningPredictionStatsToVarDict(testing_dict[i][variables_dict['target_col_index_test']], variables_dict)

    # reset kneighbors_dict
    for i in range(1, args.kneighbors + 1):
        neighbors_dict[i] = {'index' : -1, 'distance' : 1000, 'type' : ''}

PrintConfusionMatrix('knn', variables_dict)
PrintConfusionMatrix('ldf', variables_dict)
PrintConfusionMatrix('com', variables_dict)

print(f"\nall:      right |                  {variables_dict['com_knn_ldf_right']} \t| {round((variables_dict['com_knn_ldf_right']/variables_dict['test_runs_count']),2)}%")
print(f"com, knn: right | ldf:      wrong: {variables_dict['com_knn_right_ldf_wrong']} \t| {round((variables_dict['com_knn_right_ldf_wrong']/variables_dict['test_runs_count']),2)}%")
print(f"com, ldf: right | knn:      wrong: {variables_dict['com_ldf_right_knn_wrong']} \t| {round((variables_dict['com_ldf_right_knn_wrong']/variables_dict['test_runs_count']),2)}%")
print(f"knn:      right | com, ldf: wrong: {variables_dict['com_ldf_wrong_knn_right']} \t| {round((variables_dict['com_ldf_wrong_knn_right']/variables_dict['test_runs_count']),2)}%")
print(f"ldf:      right | com, knn: wrong: {variables_dict['com_knn_wrong_ldf_right']} \t| {round((variables_dict['com_knn_wrong_ldf_right']/variables_dict['test_runs_count']),2)}%")
print(f"                | all:      wrong: {variables_dict['com_knn_ldf_wrong']} \t| {round((variables_dict['com_knn_ldf_wrong']/variables_dict['test_runs_count']),2)}%")

print(f"\nldf: min:{round(variables_dict['ldf_diff_min'],2)} | max:{round(variables_dict['ldf_diff_max'],2)}")
if variables_dict['ldf_confidence_zero'] > 0:
    print(f"ldf confidence <= 0: {variables_dict['ldf_confidence_zero']} \t| {round((variables_dict['ldf_confidence_zero']/variables_dict['test_runs_count']),2)}%")

