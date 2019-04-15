########################################################################
# Jason Baumbach
#   CSC 546 - Project (due: April 06, 2019 @ 5:59 PM)
#   
#       Combined Classifier:
#           K-Nearest Neighbor (KNN) & Plugin Linear Discriminant Function (LDF)
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
parser = argparse.ArgumentParser(description="Perform k-Nearest Neighbor, plugin-LDF, and their combination to classify train and test sets of varying n-dimensional data.")
parser.add_argument("-k", "--kneighbors", type=int, choices=range(1, 30), default=3, help="number of nearest neighbors to use")
parser.add_argument("-ft", "--filetrain", default="../uci_hd_preprocessing/data/cleveland_smoke_uci+_normal_train.csv", help="training file name (and path if not in . dir)")
parser.add_argument("-fs", "--filetest", default="../uci_hd_preprocessing/data/cleveland_smoke_uci+_normal_test.csv", help="testing file name (and path if not in . dir)")
parser.add_argument("-tn", "--targetname", default="target", help="the name of the target attribute")
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2, 3], default=0, help="increase output verbosity")
args = parser.parse_args()

#   -- KNN specific --
# create a dictionary of list objects equal to the number of neighbors to test
neighbors_dict = {}
for i in range(1, args.kneighbors + 1):
    neighbors_dict[i] = {'index' : -1, 'distance' : 1000, 'type' : ''}

# debug info
if args.verbosity > 0:
    print(f"neighbors: {args.kneighbors} = {len(neighbors_dict)} :len(neighbors_dict)")

#   -- KNN specific --
# compute Euclidean distance between any two given vectors with any length.
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

#   -- LDF specific --
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
    # UCI Heart Disease specific - attribut universally ignored
    #   Note: if they exist they are represented by the pre-processing added "smoke" attribute
    , 'ignore_columns' : ['cigs', 'years']
    # variables to enable dynamic use/ignore of target attribute
    , 'target_col_name' : args.targetname
    , 'target_col_index_train' : 0
    , 'target_col_index_test' : 0
    , 'training_col_count' : 0
    , 'testing_col_count' : 0
    # algorithm variables summed and compared after testing (com = combined)
    , 'test_runs_count' : 0
    , 'com_knn_ldf_right' : 0
    , 'com_knn_right_ldf_wrong' : 0
    , 'com_ldf_right_knn_wrong' : 0
    , 'com_ldf_wrong_knn_right' : 0
    , 'com_knn_wrong_ldf_right' : 0
    , 'com_knn_ldf_wrong' : 0
    # total times LDF confidence = 0
    , 'ldf_confidence_zero' : 0
    # used to compute the min-max of LDF confidence
    , 'ldf_diff_min' : 1000
    , 'ldf_diff_max' : 0
    # initialization of the three confusion matrices
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

# Specific to UCI's Heart Disease data set which has two target columns: num (0-4) & target (0 or 1)
# Allows the code to dynamically ignore the column NOT specified in the command line
if args.targetname == 'num':
    variables_dict['ignore_columns'].append('target')
elif args.targetname == 'target':
    variables_dict['ignore_columns'].append('num')

# Read the command line specified CSV data files
def ReadFileDataIntoDictOfLists(sFileName, dDictOfLists):
    # read the file
    with open(sFileName) as csv_file:
        line_number = 0
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            dDictOfLists[line_number] = row
            line_number += 1

# dynamically determine target index for inclusion/exclusion when creating vectors for comparison
def AddTargetIndexesToVarDict(dDictOfLists, dVariables):
    # column cound can vary between train and test sets
    col_count = len(dDictOfLists[0])
    # save the column count to the specified train or test variable in the dictionary
    if dDictOfLists['type'] == 'training':
        dVariables['training_col_count'] = col_count
    elif dDictOfLists['type'] == 'testing':
        dVariables['testing_col_count'] = col_count
    else:
        # this should never happen
        print(f"Warning: type {dDictOfLists['type']} not recognized.")
    # loop through the header to find and save the target column index to the variables dictionary
    for col in range(0, col_count):
        # check if the column name matches the target name command line option
        if dDictOfLists[0][col] == dVariables['target_col_name']:
            # save the column index of the found target column name
            if dDictOfLists['type'] == 'training':
                dVariables['target_col_index_train'] = col
            elif dDictOfLists['type'] == 'testing':
                dVariables['target_col_index_test'] = col

# dynamically determine target types in the training file
def AddTargetTypesToVarDict(dTrainingData, dVariables):
    dVariables['target_types'] = {} # key = type & value = count
    # loop through the training set (ignoring the header)
    for i in range(1, len(dTrainingData) - 1):
        # check if target type has already been discovered and added to the target_types variable
        if dTrainingData[i][dVariables['target_col_index_train']] not in dVariables['target_types']:
            # set to 1 upon first discovery
            dVariables['target_types'][dTrainingData[i][dVariables['target_col_index_train']]] = 1
        else:
            # otherwise sum like instances
            dVariables['target_types'][dTrainingData[i][dVariables['target_col_index_train']]] += 1

# dynamically determine the attributes shared between the train and test sets
def AddSharedAttributesToVarDict(dTrainingData, dTestingData, dVariables):
    dVariables['shared_attributes'] = [] # list of header names identical across train and test headers
    # check which data set is larger then loop through the smaller on the outside
    if dVariables['training_col_count'] < dVariables['testing_col_count']:
        # the training set is smaller so loop through its header on the outside
        for i in dTrainingData[0]:
            # ignore the irrelevant columns hard-coded at the beginning of the program ("num" or "target" are added dynmacially) 
            if i not in dVariables['ignore_columns']:
                # loop through the testing set header
                for j in dTestingData[0]:
                    # append matching header name to the list of shared attributes
                    if i == j:
                        dVariables['shared_attributes'].append(i)
    # the testing set is smaller...
    else:
        # ...so loop through its header on the outside
        for i in dTestingData[0]:
            # ignore the irrelevant columns...
            if i not in dVariables['ignore_columns']:
                # loop through the training set header
                for j in dTrainingData[0]:
                    # append matching header name to the list of shared attributes
                    if i == j:
                        dVariables['shared_attributes'].append(i)
        
# create and return a vector of only the shared attributes (excludes target attributes)
def GetVectorOfOnlySharedAttributes(dDictOfLists, sIndex, dVariables):
    header_vec = [] # for debugging only
    return_vec = [] # vector containing only shared attributes of row values at sIndex 
    col_count = 0 # train or test column number
    # set the appropriate col_count by data set type
    if dDictOfLists['type'] == 'training':
        col_count = dVariables['training_col_count']
    elif dDictOfLists['type'] == 'testing':
        col_count = dVariables['testing_col_count']
    else:
        # this should never happen
        print(f"Warning: type {dDictOfLists['type']} not recognized.")
    # loop through the header of the passed in data set
    for i in range(0, col_count):
        # ignore the row data at the index of the training set target attribute
        if i == dVariables['target_col_index_train'] and dDictOfLists['type'] == 'training':
            continue
        # ignore the row data at the index of the testing set target attribute
        elif i == dVariables['target_col_index_test'] and dDictOfLists['type'] == 'testing':
            continue
        # loop through the shared attributes list
        for col in dVariables['shared_attributes']:
        # check if the passed in header name matches a shared attribbute
            if dDictOfLists[0][i] == col:
                # append the shared attribute value at row[sIndex] col[i]
                return_vec.append(dDictOfLists[sIndex][i])
                # store for debugging incorrectly matched attributes
                header_vec.append(col)

    # debugging info
    if args.verbosity > 2:
        print(f"shared {dDictOfLists['type']}:{header_vec}")
    
    return return_vec

#   -- KNN specific --
# Populate the k-nearest neighbors by comparing all training data with test data point
def PopulateNearestNeighborsDicOfIndexes(dNeighbors, dTrainingData, vTestData, dVariables):
    distances = []  # for debugging only (store then sort all distances for comparison to the chosen distances)
    # Loop through the training set (sans header) to find the least distance(s)
    for i in range(1, len(dTrainingData) - 1):
        # create the training vector of only the shared attributes to compare to the passed in testing vector (vTestData)
        train_vec = GetVectorOfOnlySharedAttributes(dTrainingData, i, dVariables)
        # get the Euclidean distance between the test & train vectors
        EuclideanDistance = EuclideanDistanceBetweenTwoVectors(vTestData, train_vec)
        distances.append(EuclideanDistance) # for debugging only
        # reset neighbor tracking variables
        neighbor_max_index = -1 # index of neighbor furthest away
        neighbor_max_value = -1 # value of neighbor furthest away
        # Loop through the neighbors dict so the maximum stored is always replaced first
        for j in range(1, len(dNeighbors) + 1):
            if dNeighbors[j]['distance'] > neighbor_max_value:
                neighbor_max_value = dNeighbors[j]['distance']
                neighbor_max_index = j
        # save the newest least distance over the greatest existing neighbor distance
        # compare the current Euclidean distance against the value of neighbor furthest away
        if EuclideanDistance < neighbor_max_value:
            # since current distance is less, replace neighbor with max distance with current distance info
            dNeighbors[neighbor_max_index]['index'] = i
            dNeighbors[neighbor_max_index]['distance'] = EuclideanDistance
            dNeighbors[neighbor_max_index]['type'] = dTrainingData[i][dVariables['target_col_index_train']]

    # debugging: print least k-distances from all k-distances calculated for comparison with chosen neighbors
    if args.verbosity > 2:
        distances.sort()
        print("least distances:")
        for i in range(0, len(dNeighbors)):
            print(f"min{i}:({distances[i]}) \t& neighbors:({dNeighbors[i+1]['distance']})")

#   -- KNN specific --
# Calculate majority type and confidence from the majority of nearest neighbors
def AddKNNMajorityTypeToVarDict(dNeighbors, dVariables):
    type_count_dict = {} # store key = type & value = sum of neighbors with this type
    # zero out KNN majority type tracking variables
    dVariables['knn_majority_type'] = 'UNK'
    dVariables['knn_majority_count'] = 0
    dVariables['knn_confidence'] = 0
    # loop through the target types and zero out the type_count_dict
    for key in dVariables['target_types']:
        type_count_dict[key] = 0

    # loop through the nearest neighbors and total the different type hits
    for i in range(1, len(dNeighbors) + 1):
        type_count_dict[dNeighbors[i]['type']] += 1

    # loop through the target types to set the majority info for KNN confidence calculation
    for key in type_count_dict:
        # current is better than best
        if dVariables['knn_majority_count'] < type_count_dict[key]:
            # set best to current
            dVariables['knn_majority_count'] = type_count_dict[key]
            dVariables['knn_majority_type'] = key

    # calculate confidence as (majority / k-nearest neighbors)
    dVariables['knn_confidence'] = dVariables['knn_majority_count'] / len(dNeighbors)

    # debug info
    if args.verbosity > 2:
        print(f"majority:{dVariables['knn_majority_type']}{type_count_dict}")

#   -- LDF specific --
# calculate the target type means from the training data
def AddTargetTypeMeansToVarDict(dTrainingData, dVariables):
    col_sums_dic = {} # key = target | value = sum of column values
    row_count_dic = {} # key = target | value = number of rows to divide by

    # zero out the col_sums and row_count dictionaries
    for key in dVariables['target_types']:
        col_sums_dic[key] = {}
        row_count_dic[key] = 0
        # dynamically create target mean vectors for each target type to the variables dictionary for LDF calculations
        dVariables[key] = {'ldf_mean' : []} # initialized to the empty list
        # loop thought the sared attributes list
        for col in dVariables['shared_attributes']:
            if col != dVariables['target_col_name']:
                # initialize the column sum to zero since this is a shared attribute column
                col_sums_dic[key][col] = 0

    # Loop through the training set to calculate the totals required to calculate the means
    for i in range(1, len(dTrainingData) - 1): # loop through the traing set rows
        for j in range(0, len(dTrainingData[0])): # loop through the traing set columns
            for col in dVariables['shared_attributes']: # loop through the shared columns
                # check if the column is shared
                if dTrainingData[0][j] == col:
                    # only sum the non-target columns
                    if col != dVariables['target_col_name']:
                        # sum the colum values
                        col_sums_dic[dTrainingData[i][dVariables['target_col_index_train']]][col] += float(dTrainingData[i][j])
                    # use the target column as a que to increment the row count
                    else:
                        # incrament the row count
                        row_count_dic[dTrainingData[i][dVariables['target_col_index_train']]] += 1

    # dynamically calculate the appropriate number of target means
    for key in dVariables['target_types']: # loop through the target types
        for col in col_sums_dic[key]: # loop through the columns that were summed by target
            # debug info
            if args.verbosity > 2:
                print(f"col:{col}:\t{col_sums_dic[key][col]} / {row_count_dic[key]}")
            # append the colum mean to the target type mean vector
            if row_count_dic[key] > 0:
                dVariables[key]['ldf_mean'].append(col_sums_dic[key][col] / row_count_dic[key])
            else:
                # this should never happen
                print(f"Warning: LDF mean = 0 for target:{key}")
                print(f"col:{col}:\t{col_sums_dic[key][col]} / {row_count_dic[key]}")
                dVariables[key]['ldf_mean'].append(0)

#   -- LDF specific --
# calculate the largets and second largets g(x) to determine the target and confidence of the LDF function
def AddCalculatesOfPluginLDFToVarDic(vTestData, dVariables):
    # initialize calculation variables
    dVariables['ldf_best_g'] = -1 # use -1 so best begins < least possible g(d)
    dVariables['ldf_second_best_g'] = -1 # use -1 so second best begins < least possible g(d)
    dVariables['ldf_best_target'] = 'UNK'
    dVariables['ldf_second_best_target'] = 'UNK'
    dVariables['ldf_confidence'] = 0
    ldf_diff = 0
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

    # reset the max
    if dVariables['ldf_diff_max'] < ldf_diff:
        dVariables['ldf_diff_max'] = ldf_diff
    
    # reset the min
    if dVariables['ldf_diff_min'] > ldf_diff:
        dVariables['ldf_diff_min'] = ldf_diff
    
    # use min-max to calculate confidence if min & max have been initialized
    if dVariables['ldf_diff_max'] != dVariables['ldf_diff_min']:
        dVariables['ldf_confidence'] = ((ldf_diff - dVariables['ldf_diff_min']) / (dVariables['ldf_diff_max'] - dVariables['ldf_diff_min']))
    else:
        dVariables['ldf_confidence'] = ldf_diff

    # debugging: sum all LDF confidenc <= 0
    if dVariables['ldf_confidence'] < 0:
        dVariables['ldf_confidence_zero'] += 1
        if args.verbosity > 2:
            print(f"ldf diff:{dVariables['ldf_best_g']} - {dVariables['ldf_second_best_g']}")

# sum and track confusion matrix by sPrefix (i.e., knn, ild, com) and store in the variables dictionary
def TrackConfusionMatrixSums(sTestType, sPredictionType, sPrefix, dVariables):
    # target is positive
    if int(float(sTestType)) > 0: # unfortunately, assuming the target is numeric makes the code dependant on numerical target values
        # target matches prediction
        if sTestType == sPredictionType:
            # increment true positive (TP) count
            dVariables[sPrefix + '_confusion_matrix']['TP'] += 1
        # target does not match prediction
        else:
            # increment false positive (FP) count
            dVariables[sPrefix + '_confusion_matrix']['FP'] += 1
    # target is negative
    else:
        # target matches prediction
        if sTestType == sPredictionType:
            # increment true negative (TN) count
            dVariables[sPrefix + '_confusion_matrix']['TN'] += 1
        # target does not match prediction
        else:
            # increment false negative (FN) count
            dVariables[sPrefix + '_confusion_matrix']['FN'] += 1

# simply print the confusion matrix by sPrefix (i.e., knn, ild, com) along with calculated stats
def PrintConfusionMatrix(sPrefix, dVariables):
    # print the confusion matrix
    print(f"\n{sPrefix} - Confusion Matrix:\n\tTP:{dVariables[sPrefix + '_confusion_matrix']['TP']} | FN:{dVariables[sPrefix + '_confusion_matrix']['FN']}\n\tFP:{dVariables[sPrefix + '_confusion_matrix']['FP']} | TN:{dVariables[sPrefix + '_confusion_matrix']['TN']}")
    # calculate accuracy
    dVariables[sPrefix + '_accuracy'] = round((dVariables[sPrefix + '_confusion_matrix']['TP'] + dVariables[sPrefix + '_confusion_matrix']['TN']) / (dVariables[sPrefix + '_confusion_matrix']['TP'] + dVariables[sPrefix + '_confusion_matrix']['TN'] + dVariables[sPrefix + '_confusion_matrix']['FP'] + dVariables[sPrefix + '_confusion_matrix']['FN']),2)
    # calculate error rate
    dVariables[sPrefix + '_error_rate'] = round((1 - dVariables[sPrefix + '_accuracy']),2)
    # calculate precision
    dVariables[sPrefix + '_precision'] = round(dVariables[sPrefix + '_confusion_matrix']['TP'] / (dVariables[sPrefix + '_confusion_matrix']['TP'] + dVariables[sPrefix + '_confusion_matrix']['FP']),2)
    # calculate specificity
    dVariables[sPrefix + '_specificity'] = round(dVariables[sPrefix + '_confusion_matrix']['TN'] / (dVariables[sPrefix + '_confusion_matrix']['TN'] + dVariables[sPrefix + '_confusion_matrix']['FN']),2)
    # calculate false positive rate (FPR)
    dVariables[sPrefix + '_FPR'] = round(dVariables[sPrefix + '_confusion_matrix']['FP'] / (dVariables[sPrefix + '_confusion_matrix']['TN'] + dVariables[sPrefix + '_confusion_matrix']['FN']),2)
    # print the values calculated above
    print(f"Accuracy   :{dVariables[sPrefix + '_accuracy']}")
    print(f"Error Rate :{dVariables[sPrefix + '_error_rate']}")
    print(f"Precision  :{dVariables[sPrefix + '_precision']}")
    print(f"Specificity:{dVariables[sPrefix + '_specificity']}")
    print(f"FPR        :{dVariables[sPrefix + '_FPR']}")

# keep track of the running totals of which algorithms were correct and/or incorrect
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
training_dict = {'type' : 'training'} # set the type for dynamically determining shared attributes
# read the training csv into the training dict
ReadFileDataIntoDictOfLists(variables_dict['training_file'], training_dict)

# add the target indexes of the training set to the variables dictionary
AddTargetIndexesToVarDict(training_dict, variables_dict)

# add the possible target types to the variables dictionary
AddTargetTypesToVarDict(training_dict, variables_dict)
# print debugging info
if args.verbosity > 0:
    for key in variables_dict['target_types']:
        print(f"target types {key}:{variables_dict['target_types'][key]}")

# Load the testing data
testing_dict = {'type' : 'testing'} # set the type for dynamically determining shared attributes
# read the testing csv into the testing dict
ReadFileDataIntoDictOfLists(variables_dict['testing_file'], testing_dict)

# add the target indexes of the testing set to the variables dictionary
AddTargetIndexesToVarDict(testing_dict, variables_dict)

# add the shared attributes for comparing testing data with training data to the variables dictionary
AddSharedAttributesToVarDict(training_dict, testing_dict, variables_dict)
# debugging info
if args.verbosity > 0:
    # shared attribute includes "target"
    print(f"shared attributes:{variables_dict['shared_attributes']}")
    # vector attribute includes all shared attributes except the "target" attribute
    print(f"vector attributes:{GetVectorOfOnlySharedAttributes(testing_dict, 0, variables_dict)}")

#   -- LDF specific --
# add the target type means to the variables dictionary for later use
AddTargetTypeMeansToVarDict(training_dict, variables_dict)

# loop through the target types
for key in variables_dict['target_types']:
    # calculate the inner (dot) products of the different target type means
    variables_dict[key]['ldf_mean_square'] = GetInnerProductOfTwoVectors(variables_dict[key]['ldf_mean'], variables_dict[key]['ldf_mean'])

# debugging info
if args.verbosity > 1:
    for key in variables_dict['target_types']:
        print(f"{key} mean_sq:{variables_dict[key]['ldf_mean_square']} | mean:{variables_dict[key]['ldf_mean']}")

# debugging info
if args.verbosity > 0:
    # print the trainging and testing data lengths (subtract 1 for headers and 1 for starting from zero)
    print(f"train:{len(training_dict)-2} | test:{len(testing_dict)-2}")

# debugging info
if args.verbosity > 2:
    # Print some of the rows from input files
    print(f"The first 2 training samples with target:{training_dict[0][variables_dict['target_col_index_train']]}:")
    for i in range(0, 2):
        print(f"\t{i} {training_dict[i]}")

    print(f"\nThe first 2 testing samples with target:{testing_dict[0][variables_dict['target_col_index_test']]}")
    for i in range(0, 2):
        print(f"\t{i} {testing_dict[i]}")

# loop through all testing data
for i in range(1, len(testing_dict) - 1):
    # create the test vector at the i-th row from only the shared test & train attributes
    test_vec = GetVectorOfOnlySharedAttributes(testing_dict, i, variables_dict)

    # set the k-nearest neighbors in the neighbors dict
    PopulateNearestNeighborsDicOfIndexes(neighbors_dict, training_dict, test_vec, variables_dict)

    # calculate and set the KNN predicted target and confidence in the variables dict
    AddKNNMajorityTypeToVarDict(neighbors_dict, variables_dict)

    # calculate and set the LDF predicted target and confidence in the variables dict
    AddCalculatesOfPluginLDFToVarDic(test_vec, variables_dict)

    # ----- Store the Confusion Matrix running counts -----
    # track KNN confusion matrix running totals
    TrackConfusionMatrixSums(testing_dict[i][variables_dict['target_col_index_test']], variables_dict['knn_majority_type'], 'knn', variables_dict)
    # track LDF confusion matrix running totals
    TrackConfusionMatrixSums(testing_dict[i][variables_dict['target_col_index_test']], variables_dict['ldf_best_target'], 'ldf', variables_dict)

    # combine and store the confusion matrix counts for most confident prediction
    # KNN confidence > LDF confidence
    if variables_dict['knn_confidence'] > variables_dict['ldf_confidence']:
        # set the combined best target = to the KNN predicted target
        variables_dict['com_best_target'] = variables_dict['knn_majority_type']
        # debugging info
        if args.verbosity > 1:
            if variables_dict['ldf_best_target'] != variables_dict['knn_majority_type']:
                print(f"KNN:{variables_dict['knn_majority_type']}>({testing_dict[i][variables_dict['target_col_index_test']]}:confidence) KNN:{variables_dict['knn_majority_type']}:{round(variables_dict['knn_confidence'],2)} | LDF:{variables_dict['ldf_best_target']}:{round(variables_dict['ldf_confidence'],2)}")
    # LDF confidence > KNN confidence
    else:
        # set the combined best target = to the LDF predicted target
        variables_dict['com_best_target'] = variables_dict['ldf_best_target']
        # debugging infoi
        if args.verbosity > 1:
            if variables_dict['ldf_best_target'] != variables_dict['knn_majority_type']:
                print(f"LDF:{variables_dict['com_best_target']}>({testing_dict[i][variables_dict['target_col_index_test']]}:confidence) KNN:{variables_dict['knn_majority_type']}:{round(variables_dict['knn_confidence'],2)} | LDF:{variables_dict['ldf_best_target']}:{round(variables_dict['ldf_confidence'],2)}")
    
    # track Combined confusion matrix running totals
    TrackConfusionMatrixSums(testing_dict[i][variables_dict['target_col_index_test']], variables_dict['com_best_target'], 'com', variables_dict)
    # -----------------------------------------------------

    # add the running totals of predictions for KNN, LDF, & Combined
    AddRunningPredictionStatsToVarDict(testing_dict[i][variables_dict['target_col_index_test']], variables_dict)

    # reset kneighbors_dict
    for i in range(1, args.kneighbors + 1):
        neighbors_dict[i] = {'index' : -1, 'distance' : 1000, 'type' : ''}

# print the three confusion matrices
PrintConfusionMatrix('knn', variables_dict)
PrintConfusionMatrix('ldf', variables_dict)
PrintConfusionMatrix('com', variables_dict)

# print the prediction stats for KNN, LDF, & Combined
print(f"\nall:      right |                  {variables_dict['com_knn_ldf_right']} \t| {round((variables_dict['com_knn_ldf_right']/variables_dict['test_runs_count']),2)}%")
print(f"com, knn: right | ldf:      wrong: {variables_dict['com_knn_right_ldf_wrong']} \t| {round((variables_dict['com_knn_right_ldf_wrong']/variables_dict['test_runs_count']),2)}%")
print(f"com, ldf: right | knn:      wrong: {variables_dict['com_ldf_right_knn_wrong']} \t| {round((variables_dict['com_ldf_right_knn_wrong']/variables_dict['test_runs_count']),2)}%")
print(f"knn:      right | com, ldf: wrong: {variables_dict['com_ldf_wrong_knn_right']} \t| {round((variables_dict['com_ldf_wrong_knn_right']/variables_dict['test_runs_count']),2)}%")
print(f"ldf:      right | com, knn: wrong: {variables_dict['com_knn_wrong_ldf_right']} \t| {round((variables_dict['com_knn_wrong_ldf_right']/variables_dict['test_runs_count']),2)}%")
print(f"                | all:      wrong: {variables_dict['com_knn_ldf_wrong']} \t| {round((variables_dict['com_knn_ldf_wrong']/variables_dict['test_runs_count']),2)}%")

# print LDF min & max values for reference
print(f"\nldf: min:{round(variables_dict['ldf_diff_min'],2)} | max:{round(variables_dict['ldf_diff_max'],2)}")

# debugging info - print LDF confidence == 0 summation
if variables_dict['ldf_confidence_zero'] > 0:
    print(f"ldf confidence <= 0: {variables_dict['ldf_confidence_zero']} \t| {round((variables_dict['ldf_confidence_zero']/variables_dict['test_runs_count']),2)}%")
