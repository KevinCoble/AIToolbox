//
//  MachineLearningProtocols.swift
//  AIToolbox
//
//  Created by Kevin Coble on 1/16/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation

public protocol Classifier {
    func trainClassifier(trainData: DataSet) throws
    func classifyOne(inputs: [Double]) throws ->Int
    func classify(testData: DataSet) throws
}

public protocol Regressor {
    func trainRegressor(trainData: DataSet) throws
    func predictOne(inputs: [Double]) throws ->[Double]
    func predict(testData: DataSet) throws
}

public protocol NonLinearEquation {
    //  If output dimension > 1, parameters is a matrix with each row the parameters for one of the outputs
    var parameters: [Double] { get set }

    func getInputDimension() -> Int
    func getOutputDimension() -> Int
    func getParameterDimension() -> Int     //  This must be an integer multiple of output dimension
    func getOutputs(inputs: [Double]) throws -> [Double]        //  Returns vector outputs sized for outputs
    func getGradient(inputs: [Double]) throws -> [Double]       //  Returns vector gradient sized for parameters - can be stubbed for ParameterDelta method
}

extension Classifier {
    ///  Calculate the precentage correct on a classification network using a test data set
    public func getClassificationPercentage(testData: DataSet) throws -> Double
    {
        //  Verify the data set is the right type
        if (testData.dataType != .Classification) { throw DataTypeError.InvalidDataType }
        
        var countCorrect = 0
        
        //  Do for the entire test set
        for index in 0..<testData.size {
            //  Get the results of a feedForward run
            let result = try classifyOne(testData.inputs[index])
            if (result == testData.classes![index]) {countCorrect += 1}
        }
        
        //  Calculate the percentage
        return Double(countCorrect) / Double(testData.size)
    }
}

extension Regressor {
    
    ///  Calculate the total absolute value of error on a regressor using a test data set
    public func getTotalAbsError(testData: DataSet) throws -> Double
    {
        //  Verify the data set is the right type
        if (testData.dataType != .Regression) { throw DataTypeError.InvalidDataType }
        
        var sum = 0.0
        
        //  Do for the entire test set
        for index in 0..<testData.size{
            //  Get the results of a prediction
            let results = try predictOne(testData.inputs[index])
            
            //  Sum up the differences
            for nodeIndex in 0..<results.count {
                sum += abs(results[nodeIndex] - testData.outputs![index][nodeIndex])
            }
            
        }
        
        return sum
    }
    
}
