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
            if args.verbosity > 1:
                print(f"vOne[{i}]:{vOne[i]} * vTwo[{i}]:{vTwo[i]}")
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
    , 'majority_type' : 0
    , 'classification' : 'UNK'
    , 'ignore_columns' : ['cigs', 'years', 'num']
    , 'confusion_matrix' : {
        'TN'    : 0
        , 'FP'  : 0
        , 'FN'  : 0
        , 'TP'  : 0
    }
}

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
def GetNearestNeighborMajorityType(dNeighbors, dVariables):
    type_count_dict = {}
    the_majority_type = -1
    current_max_count = 0
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
        if current_max_count < type_count_dict[key]:
            current_max_count = type_count_dict[key]
            the_majority_type = key

    if args.verbosity > 1:
        print(f"majority:{the_majority_type}{type_count_dict}")
        # for key, value in type_count_dict.items():
        #     print(f"{key}:{value}")

    return the_majority_type

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
            if args.verbosity > 0:
                print(f"col:{col}:\t{col_sums_dic[key][col]} / {row_count_dic[key]}")
            # append the colum mean to the target type mean vector
            dVariables[key]['ldf_mean'].append(col_sums_dic[key][col] / row_count_dic[key]) 

def TrackConfusionMatrixSums(sTestType, sPredictionType, sPrefix, dVariables):
    # store the confusion matrix counts (unfortunately this makes the code dependant on numerical target values)
    if int(float(sTestType)) > 0:
        if sTestType == sPredictionType:
            variables_dict[sPrefix + 'classification'] = 'Correct'
            variables_dict[sPrefix + 'confusion_matrix']['TP'] += 1
        else:
            variables_dict[sPrefix + 'classification'] = 'Incorrect'
            variables_dict[sPrefix + 'confusion_matrix']['FP'] += 1
    else:
        if sTestType == sPredictionType:
            variables_dict[sPrefix + 'classification'] = 'Correct'
            variables_dict[sPrefix + 'confusion_matrix']['TN'] += 1
        else:
            variables_dict[sPrefix + 'classification'] = 'Incorrect'
            variables_dict[sPrefix + 'confusion_matrix']['FN'] += 1

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

    # get the k-nearest neighbors' majority type
    variables_dict['majority_type'] = GetNearestNeighborMajorityType(neighbors_dict, variables_dict)

    # get the largest PluginLDF value of g(x)
    variables_dict['ldf_best_g'] = -1
    variables_dict['ldf_best_target'] = -1
    # loop through the target types
    for key in variables_dict['target_types']:
        # calculate the inner (dot) products of the target type means
        variables_dict[key]['ldf_dot_mean'] = GetInnerProductOfTwoVectors(test_vec, variables_dict[key]['ldf_mean'])
        # calculate g(x)
        variables_dict[key]['ldf_g'] = (2 * variables_dict[key]['ldf_dot_mean']) - variables_dict[key]['ldf_mean_square']
        # store the largest g(x)
        if variables_dict['ldf_best_g'] < variables_dict[key]['ldf_g']:
            variables_dict['ldf_best_g'] = variables_dict[key]['ldf_g']
            variables_dict['ldf_best_target'] = key
        # debug info: print the formul used
        if args.verbosity > 1:
            print(f"\t{key}_g(x): {round(variables_dict[key]['ldf_g'], 2)} = (2 * {round(variables_dict[key]['ldf_dot_mean'], 2)}) - {round(variables_dict[key]['ldf_mean_square'], 2)}")

    # store the confusion matrix counts (unfortunately this makes the code dependant on numerical target values)
    if int(float(testing_dict[i][variables_dict['target_col_index_test']])) > 0:
        if testing_dict[i][variables_dict['target_col_index_test']] == variables_dict['majority_type']:
#        if testing_dict[i][variables_dict['target_col_index_test']] == variables_dict['ldf_best_target']:
            variables_dict['classification'] = 'Correct'
            variables_dict['confusion_matrix']['TP'] += 1
        else:
            variables_dict['classification'] = 'Incorrect'
            variables_dict['confusion_matrix']['FP'] += 1
    else:
        if testing_dict[i][variables_dict['target_col_index_test']] == variables_dict['majority_type']:
#        if testing_dict[i][variables_dict['target_col_index_test']] == variables_dict['ldf_best_target']:
            variables_dict['classification'] = 'Correct'
            variables_dict['confusion_matrix']['TN'] += 1
        else:
            variables_dict['classification'] = 'Incorrect'
            variables_dict['confusion_matrix']['FN'] += 1

    if args.verbosity > 0:
        print(f"{i+1} - Test type {testing_dict[i][variables_dict['target_col_index_test']]} & LDF:{variables_dict['ldf_best_target']} & Most neighbors {variables_dict['majority_type']}: {variables_dict['classification']}")

    # reset variables
    variables_dict['majority_type'] = 0
    variables_dict['classification'] = 'UNK'
    for i in range(1, args.kneighbors + 1):
        neighbors_dict[i] = {'index' : -1, 'distance' : 1000, 'type' : ''}

print(f"Confusion Matrix:\n\tTP:{variables_dict['confusion_matrix']['TP']} | FN:{variables_dict['confusion_matrix']['FN']}\n\tFP:{variables_dict['confusion_matrix']['FP']} | TN:{variables_dict['confusion_matrix']['TN']}")
accuracy = round((variables_dict['confusion_matrix']['TP'] + variables_dict['confusion_matrix']['TN']) / (variables_dict['confusion_matrix']['TP'] + variables_dict['confusion_matrix']['TN'] + variables_dict['confusion_matrix']['FP'] + variables_dict['confusion_matrix']['FN']),2)
error_rate = round((1-accuracy),2)
print(f"Accuracy   :{accuracy}")
print(f"Error Rate :{error_rate}")
print(f"Precision  :{round(variables_dict['confusion_matrix']['TP'] / (variables_dict['confusion_matrix']['TP'] + variables_dict['confusion_matrix']['FP']),2)}")
print(f"Specificity:{round(variables_dict['confusion_matrix']['TN'] / (variables_dict['confusion_matrix']['TN'] + variables_dict['confusion_matrix']['FN']),2)}")
print(f"FPR        :{round(variables_dict['confusion_matrix']['FP'] / (variables_dict['confusion_matrix']['TN'] + variables_dict['confusion_matrix']['FN']),2)}")
#print(f"Accuracy:{(variables_dict['confusion_matrix']['TP'] + variables_dict['confusion_matrix']['TN']) / (variables_dict['confusion_matrix']['TP'] + variables_dict['confusion_matrix']['TN'] + variables_dict['confusion_matrix']['FP'] + variables_dict['confusion_matrix']['FN'])}")
# for key, value in variables_dict['confusion_matrix'].items():
#     print(f"{key}:{value}")