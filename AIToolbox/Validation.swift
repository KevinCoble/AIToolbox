//
//  Validation.swift
//  AIToolbox
//
//  Created by Kevin Coble on 4/24/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation

public enum ValidationError: Error {
    case computationError
}

///  Class for using validation techniques for model/parameter selection
open class Validation
{
    let validationType : DataSetType
    var classifiers : [Classifier] = []
    var regressors : [Regressor] = []
    var validationTestResults : [Double] = []    //  Array of average scores on tests for each model
    var timeout: Int64 = 1000000000 * 60 * 60       //  In nanoseconds, default of 1 hour
    
    public init(type: DataSetType)
    {
        validationType = type
    }
    
    open func addModel(_ model: Classifier) throws
    {
        if (validationType == .regression) { throw MachineLearningError.modelNotRegression }
        
        classifiers.append(model)
    }
    
    open func addModel(_ model: Regressor) throws
    {
        if (validationType == .classification) { throw MachineLearningError.modelNotClassification }
        
        regressors.append(model)
    }
    
    ///  Method to split data into train and test sets, then do basic validation
    ///  Returns model index that did the best.  Scores in validationTestResults
    ///  Model should probably be re-trained on all data afterwards
    ///  Uses Grand-Central-Dispatch to run in parallel
    open func simpleValidation(_ data: DataSet, fractionForTest: Double) throws -> Int
    {
        //  Get the size of the training and testing sets
        let testSize = Int(fractionForTest * Double(data.size))
        if (testSize < 1) { throw MachineLearningError.notEnoughData }
        if (testSize == data.size) { throw MachineLearningError.notEnoughData }
        
        //  Split the data into a training and a testing set
        let indices = data.getRandomIndexSet()
        let testData = DataSet(fromDataSet: data, withEntries: indices[0..<testSize])!
        let trainData = DataSet(fromDataSet: data, withEntries: indices[testSize..<data.size])!
        
        //  Get a concurrent GCD queue to run this all in
        let queue = DispatchQueue.global(qos: DispatchQoS.QoSClass.default)
        
        //  Get a GCD group so we can know when all models are done
        let group = DispatchGroup();
        
        //  Train and test each model
        if (validationType == .regression) {
            validationTestResults = [Double](repeating: 0.0, count: regressors.count)
            for modelIndex in 0..<regressors.count {
                queue.async(group: group) {
                    do {
                        try self.regressors[modelIndex].trainRegressor(trainData)
                        self.validationTestResults[modelIndex] = try self.regressors[modelIndex].getTotalAbsError(testData)
                    }
                    catch {
                        self.validationTestResults[modelIndex] = Double.infinity        //  failure needs to be checked later
                    }
                }
            }
            
            //  Wait for the models to finish calculating
            let timeoutTime = DispatchTime.now() + Double(timeout) / Double(NSEC_PER_SEC)
            if (group.wait(timeout: timeoutTime) != DispatchTimeoutResult.success) {
                throw MachineLearningError.operationTimeout
            }
            
            //  Rescale errors to have minimum error = 1.0
            var minError = Double.infinity
            for error in validationTestResults {
                if (error == Double.infinity) { throw ValidationError.computationError }
                if (error < minError) { minError = error }
            }
            for modelIndex in 0..<regressors.count {
                if (validationTestResults[modelIndex] == 0.0) {
                    validationTestResults[modelIndex] = 1.0
                }
                else {
                    validationTestResults[modelIndex] = 1.0 - ((validationTestResults[modelIndex] - minError) / validationTestResults[modelIndex])
                }
            }
        }
        else {
            validationTestResults = [Double](repeating: 0.0, count: classifiers.count)
            for modelIndex in 0..<classifiers.count {
                queue.async(group: group) {
                    do {
                        try self.classifiers[modelIndex].trainClassifier(trainData)
                        self.validationTestResults[modelIndex] = try self.classifiers[modelIndex].getClassificationPercentage(testData)
                    }
                    catch {
                        self.validationTestResults[modelIndex] = Double.infinity        //  failure needs to be checked later
                    }
                }
            }
            
            //  Wait for the models to finish calculating
            let timeoutTime = DispatchTime.now() + Double(timeout) / Double(NSEC_PER_SEC)
            if (group.wait(timeout: timeoutTime) != DispatchTimeoutResult.success) {
                throw MachineLearningError.operationTimeout
            }
            
            //  Check for an error during multithreading
            for error in validationTestResults {
                if (error == Double.infinity) { throw ValidationError.computationError }
            }
        }
        
        //  Find the model that did best
        var bestResult = -Double.infinity
        var bestIndex = -1
        for modelIndex in 0..<validationTestResults.count {
            if (validationTestResults[modelIndex] > bestResult) {
                bestResult = validationTestResults[modelIndex]
                bestIndex = modelIndex
            }
        }
        return bestIndex
    }
    
    ///  Method to split data into N parcels, then train on N-1 parcels, test on last parcel, then repeat for all permutations
    ///  If N is set to dataset size, this turns into 'leave-one-out' cross validation
    ///  Returns model that did the best.  Model should probably be re-trained on all data afterwards
    ///  Uses Grand-Central-Dispatch to run in parallel
    open func NFoldCrossValidation(_ data: DataSet, numberOfFolds: Int) throws -> Int
    {
        if (numberOfFolds > data.size) { throw MachineLearningError.notEnoughData }
        
        //  Get a concurrent GCD queue to run this all in
        let queue = DispatchQueue.global(qos: DispatchQoS.QoSClass.default)
        
        //  Get a GCD group so we can know when all models are done
        let group = DispatchGroup();

        //  Randomize the points into the different folds
        let indices = data.getRandomIndexSet()
        var startIndex = 0
        
        //  Initialize the results to 0
        validationTestResults = [Double](repeating: 0.0, count: regressors.count)
        
        //  Do for each fold
        for fold in 0..<numberOfFolds {
            //  Get the number of points in this fold
            let foldSize = (data.size - startIndex) / (numberOfFolds - fold)
            
            //  Split the data into a training and a testing set
            let trainData = DataSet(dataType : data.dataType, inputDimension : data.inputDimension, outputDimension : data.outputDimension)
            if (fold != 0) {
                try trainData.includeEntries(fromDataSet: data, withEntries: indices[0..<startIndex])
            }
            let testData = DataSet(fromDataSet: data, withEntries: indices[startIndex..<startIndex+foldSize])!
            if (fold != numberOfFolds-1) {
                try trainData.includeEntries(fromDataSet: data, withEntries: indices[startIndex+foldSize..<data.size])
            }
            startIndex += foldSize
            
            //  Train and test each model with this fold
            if (validationType == .regression) {
                for modelIndex in 0..<regressors.count {
                    queue.async(group: group) {
                        do {
                            try self.regressors[modelIndex].trainRegressor(trainData)
                            self.validationTestResults[modelIndex] += try self.regressors[modelIndex].getTotalAbsError(testData)
                        }
                        catch {
                            self.validationTestResults[modelIndex] = Double.infinity        //  failure needs to be checked later
                        }
                    }
                }
            }
            else {
                for modelIndex in 0..<classifiers.count {
                    queue.async(group: group) {
                        do {
                            try self.classifiers[modelIndex].trainClassifier(trainData)
                            self.validationTestResults[modelIndex] += try self.classifiers[modelIndex].getClassificationPercentage(testData)
                        }
                        catch {
                            self.validationTestResults[modelIndex] = Double.infinity        //  failure needs to be checked later
                        }
                    }
                }
            }
            
            //  Wait for the models to finish calculating before the next fold
            let timeoutTime = DispatchTime.now() + Double(timeout) / Double(NSEC_PER_SEC)
            if (group.wait(timeout: timeoutTime) != DispatchTimeoutResult.success) {
                throw MachineLearningError.operationTimeout
            }
        }
        
        //  Check for an error during multithreading
        for error in validationTestResults {
            if (error == Double.infinity) { throw ValidationError.computationError }
        }
        
        //  Scale the validation results
        if (validationType == .regression) {
            var minError = Double.infinity
            for error in validationTestResults {
                if (error == Double.infinity) { throw ValidationError.computationError }
                if (error < minError) { minError = error }
            }
            for modelIndex in 0..<regressors.count {
                if (validationTestResults[modelIndex] == 0.0) {
                    validationTestResults[modelIndex] = 1.0
                }
                else {
                    validationTestResults[modelIndex] = 1.0 - ((validationTestResults[modelIndex] - minError) / validationTestResults[modelIndex])
                }
            }
        }
        else {
            for modelIndex in 0..<classifiers.count {
                validationTestResults[modelIndex] /= Double(numberOfFolds)
            }
        }
        
        //  Find the model that did best
        var bestResult = -Double.infinity
        var bestIndex = -1
        for modelIndex in 0..<validationTestResults.count {
            if (validationTestResults[modelIndex] > bestResult) {
                bestResult = validationTestResults[modelIndex]
                bestIndex = modelIndex
            }
        }
        return bestIndex
    }
}
