//
//  MachineLearningProtocols.swift
//  AIToolbox
//
//  Created by Kevin Coble on 1/16/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation

public enum MachineLearningError: Error {
    case dataNotRegression
    case dataWrongDimension
    case notEnoughData
    case modelNotRegression
    case modelNotClassification
    case notTrained
    case initializationError
    case didNotConverge
    case continuationNotSupported
    case operationTimeout
    case continueTrainingClassesNotSame
}


public protocol Classifier {
    func getInputDimension() -> Int
    func getParameterDimension() -> Int     //  May only be valid after training
    func getNumberOfClasses() -> Int        //  May only be valid after training
    func setParameters(_ parameters: [Double]) throws
    func setCustomInitializer(_ function: ((_ trainData: DataSet)->[Double])!)
    func getParameters() throws -> [Double]
    func trainClassifier(_ trainData: DataSet) throws
    func continueTrainingClassifier(_ trainData: DataSet) throws      //  Trains without initializing parameters first
    func classifyOne(_ inputs: [Double]) throws ->Int
    func classify(_ testData: DataSet) throws
}

public protocol Regressor {
    func getInputDimension() -> Int
    func getOutputDimension() -> Int
    func getParameterDimension() -> Int
    func setParameters(_ parameters: [Double]) throws
    func setCustomInitializer(_ function: ((_ trainData: DataSet)->[Double])!)
    func getParameters() throws -> [Double]
    func trainRegressor(_ trainData: DataSet) throws
    func continueTrainingRegressor(_ trainData: DataSet) throws      //  Trains without initializing parameters first
    func predictOne(_ inputs: [Double]) throws ->[Double]
    func predict(_ testData: DataSet) throws
}

public protocol NonLinearEquation {
    //  If output dimension > 1, parameters is a matrix with each row the parameters for one of the outputs
    var parameters: [Double] { get set }

    func getInputDimension() -> Int
    func getOutputDimension() -> Int
    func getParameterDimension() -> Int     //  This must be an integer multiple of output dimension
    func setParameters(_ parameters: [Double]) throws
    func getOutputs(_ inputs: [Double]) throws -> [Double]        //  Returns vector outputs sized for outputs
    func getGradient(_ inputs: [Double]) throws -> [Double]       //  Returns vector gradient sized for parameters - can be stubbed for ParameterDelta method
}

extension Classifier {
    ///  Calculate the precentage correct on a classification network using a test data set
    public func getClassificationPercentage(_ testData: DataSet) throws -> Double
    {
        //  Verify the data set is the right type
        if (testData.dataType != .classification) { throw DataTypeError.invalidDataType }
        
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
    public func getTotalAbsError(_ testData: DataSet) throws -> Double
    {
        //  Verify the data set is the right type
        if (testData.dataType != .regression) { throw DataTypeError.invalidDataType }
        
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

internal class ClassificationData {
    var foundLabels: [Int] = []
    var classCount: [Int] = []
    var classOffsets: [[Int]] = []
    
    var numClasses: Int
    {
        return foundLabels.count
    }
}

