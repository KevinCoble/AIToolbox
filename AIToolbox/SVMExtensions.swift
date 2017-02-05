//
//  SVMExtensions.swift
//  AIToolbox
//
//  Created by Kevin Coble on 1/17/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//
//  This file contains extensions to the SVMModel class to get it to play nicely with the rest of the library
//  This code doesn't go in the SVM file, as I want to keep that close to the original LIBSVM source material

import Foundation


enum SVMError: Error {
    case invalidModelType
}


extension SVMModel : Classifier {
    public func getInputDimension() -> Int
    {
        if (supportVector.count < 1) { return 0 }
        return supportVector[0].count
    }
    public func getParameterDimension() -> Int
    {
        return totalSupportVectors      //!!  This needs to be calculated correctly
    }
    public func getNumberOfClasses() -> Int
    {
        return numClasses
    }
    public func setParameters(_ parameters: [Double]) throws
    {
        //!!   This needs to be filled in
    }
    public func getParameters() throws  -> [Double]
    {
        //!!   This needs to be filled in
        return []
    }
    
    public func setCustomInitializer(_ function: ((_ trainData: MLDataSet)->[Double])!) {
        //  Ignore, as SVM doesn't use an initialization
    }

    public func trainClassifier(_ trainData: MLClassificationDataSet) throws
    {
        //  Verify the SVMModel is the right type
        if type != .c_SVM_Classification && type != .ν_SVM_Classification { throw SVMError.invalidModelType }
        
        //  Verify the data set is the right type
        if (trainData.dataType == .regression) { throw DataTypeError.invalidDataType }
        
        //  Train on the data (ignore initialization, as SVM's do single-batch training)
        if (trainData is DataSet) {
            train(trainData as! DataSet)
        }
        else {
            //  Convert the data set to a DataSet class, as the SVM was ported from a public domain code that used specific properties that were added to the DataSet class but are not in the MLDataSet protocols
            if let convertedData = DataSet(fromClassificationDataSet: trainData) {
                train(convertedData)
            }
        }
    }
    
    public func continueTrainingClassifier(_ trainData: MLClassificationDataSet) throws
    {
        //  Linear regression uses one-batch training (solved analytically)
        throw MachineLearningError.continuationNotSupported
    }
    
    public func classifyOne(_ inputs: [Double]) ->Int
    {
        //  Get the support vector start index for each class
        var coeffStart = [0]
        for index in 0..<numClasses-1 {
            coeffStart.append(coeffStart[index] + supportVectorCount[index])
        }
        //  Get the kernel value for each support vector at the input value
        var kernelValue: [Double] = []
        for sv in 0..<totalSupportVectors {
            kernelValue.append(Kernel.calcKernelValue(kernelParams, x: inputs, y: supportVector[sv]))
        }
        
        //  Allocate vote space for the classification
        var vote = [Int](repeating: 0, count: numClasses)
        
        //  Initialize the decision values
        var decisionValues: [Double] = []
        
        //  Get the seperation info between each class pair
        var permutation = 0
        for i in 0..<numClasses {
            for j in i+1..<numClasses {
                var sum = 0.0
                for k in 0..<supportVectorCount[i] {
                    sum += coefficients[j-1][coeffStart[i]+k] * kernelValue[coeffStart[i]+k]
                }
                for k in 0..<supportVectorCount[j] {
                    sum += coefficients[i][coeffStart[j]+k] * kernelValue[coeffStart[j]+k]
                }
                sum -= ρ[permutation]
                decisionValues.append(sum)
                permutation += 1
                if (sum > 0) {
                    vote[i] += 1
                }
                else {
                    vote[j] += 1
                }
            }
        }
        
        //  Get the most likely class, and set it
        var maxIndex = 0
        for index in 1..<numClasses {
            if (vote[index] > vote[maxIndex]) { maxIndex = index }
        }
        
        return labels[maxIndex]
    }
    
    public func classify(_ testData: MLClassificationDataSet) throws
    {
        //  Verify the SVMModel is the right type
        if type != .c_SVM_Classification || type != .ν_SVM_Classification { throw SVMError.invalidModelType }
        
        //  Verify the data set is the right type
        if (testData.dataType != .classification) { throw DataTypeError.invalidDataType }
        if (supportVector.count <= 0) { throw MachineLearningError.notTrained }
        if (testData.inputDimension != supportVector[0].count) { throw DataTypeError.wrongDimensionOnInput }
        
        //  Put the data into a DataSet for SVM (it uses a DataSet so that it can be both regressor and classifier)
        if let data = DataSet(dataType: .classification, withInputsFrom: testData) {
        
            //  Predict
            predictValues(data)
        
            //  Transfer the predictions back to the classifier data set
            for index in 0..<testData.size {
                let resultClass = try data.getClass(index)
                try testData.setClass(index, newClass: resultClass)
            }
        }
        else {
            throw MachineLearningError.dataWrongDimension
        }
    }
}
