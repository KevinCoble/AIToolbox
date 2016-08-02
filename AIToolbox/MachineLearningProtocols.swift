//
//  MachineLearningProtocols.swift
//  AIToolbox
//
//  Created by Kevin Coble on 1/16/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation

public enum MachineLearningError: ErrorType {
    case DataNotRegression
    case DataNotClassification
    case DataWrongDimension
    case NotEnoughData
    case ModelNotRegression
    case ModelNotClassification
    case NotTrained
    case InitializationError
    case DidNotConverge
    case ContinuationNotSupported
    case OperationTimeout
    case ContinueTrainingClassesNotSame
}


public enum DataSetType   //  data type
{
    case Regression
    case Classification
    case RealAndClass           //  Has both real array and class (SVM predict values, etc.)
}

public protocol MLDataSet : class {     //  Machine learning data set provider
    var dataType : DataSetType { get }
    var inputDimension: Int  { get }
    var outputDimension: Int { get }
    var size: Int { get }
    var optionalData: AnyObject? { get set }      //  Optional data that can be temporarily added by methods using the data set
    
    func getInput(index: Int) throws ->[Double]
}

public protocol MLRegressionDataSet : MLDataSet {     //  Machine learning regression data set provider
    func getOutput(index: Int) throws ->[Double]
    
    func setOutput(index: Int, newOutput : [Double]) throws
}

public protocol MLClassificationDataSet : MLDataSet {     //  Machine learning classification data set provider
    func getClass(index: Int) throws ->Int
    
    func setClass(index: Int, newClass : Int) throws
}

public protocol MLCombinedDataSet : MLRegressionDataSet, MLClassificationDataSet {     //  Machine learning classification AND regression data set provider (for classes that can do both)
}

public protocol Classifier {
    func getInputDimension() -> Int
    func getParameterDimension() -> Int     //  May only be valid after training
    func getNumberOfClasses() -> Int        //  May only be valid after training
    func setParameters(parameters: [Double]) throws
    func setCustomInitializer(function: ((trainData: MLDataSet)->[Double])!)
    func getParameters() throws -> [Double]
    func trainClassifier(trainData: MLClassificationDataSet) throws
    func continueTrainingClassifier(trainData: MLClassificationDataSet) throws      //  Trains without initializing parameters first
    func classifyOne(inputs: [Double]) throws ->Int
    func classify(testData: MLClassificationDataSet) throws
}

public protocol Regressor {
    func getInputDimension() -> Int
    func getOutputDimension() -> Int
    func getParameterDimension() -> Int
    func setParameters(parameters: [Double]) throws
    func setCustomInitializer(function: ((trainData: MLDataSet)->[Double])!)
    func getParameters() throws -> [Double]
    func trainRegressor(trainData: MLRegressionDataSet) throws
    func continueTrainingRegressor(trainData: MLRegressionDataSet) throws      //  Trains without initializing parameters first
    func predictOne(inputs: [Double]) throws ->[Double]
    func predict(testData: MLRegressionDataSet) throws
}

public protocol NonLinearEquation {
    //  If output dimension > 1, parameters is a matrix with each row the parameters for one of the outputs
    var parameters: [Double] { get set }

    func getInputDimension() -> Int
    func getOutputDimension() -> Int
    func getParameterDimension() -> Int     //  This must be an integer multiple of output dimension
    func setParameters(parameters: [Double]) throws
    func getOutputs(inputs: [Double]) throws -> [Double]        //  Returns vector outputs sized for outputs
    func getGradient(inputs: [Double]) throws -> [Double]       //  Returns vector gradient sized for parameters - can be stubbed for ParameterDelta method
}

extension MLDataSet {
    public func getRandomIndexSet() -> [Int]
    {
        //  Get the ordered array of indices
        var shuffledArray: [Int] = []
        for i in 0..<size { shuffledArray.append(i) }
        
        // empty and single-element collections don't shuffle
        if size < 2 { return shuffledArray }
        
        //  Shuffle
        for i in 0..<size - 1 {
            let j = Int(arc4random_uniform(UInt32(size - i))) + i
            guard i != j else { continue }
            swap(&shuffledArray[i], &shuffledArray[j])
        }
        
        return shuffledArray
    }
}

extension MLRegressionDataSet {
    
    public func getInputRange() -> [(minimum: Double, maximum: Double)]
    {
        //  Allocate the array of tuples
        var results : [(minimum: Double, maximum: Double)] = Array(count: inputDimension, repeatedValue: (minimum: Double.infinity, maximum: -Double.infinity))
        
        //  Go through each input
        for index in 0..<size {
            do {
                let input = try getInput(index)
                //  Go through each dimension
                for dimension in 0..<inputDimension {
                    if (input[dimension] < results[dimension].minimum) { results[dimension].minimum = input[dimension] }
                    if (input[dimension] > results[dimension].maximum) { results[dimension].maximum = input[dimension] }
                }
            }
            catch {
                //  Error getting input array
            }
        }
        
        return results
    }
    
    public func getOutputRange() -> [(minimum: Double, maximum: Double)]
    {
        //  Allocate the array of tuples
        var results : [(minimum: Double, maximum: Double)] = Array(count: outputDimension, repeatedValue: (minimum: Double.infinity, maximum: -Double.infinity))
        
        //  Go through each output
        for index in 0..<size {
            do {
                let outputs = try getOutput(index)
                //  Go through each dimension
                for dimension in 0..<outputDimension {
                    if (outputs[dimension] < results[dimension].minimum) { results[dimension].minimum = outputs[dimension] }
                    if (outputs[dimension] > results[dimension].maximum) { results[dimension].maximum = outputs[dimension] }
                }
            }
            catch {
                //  Error getting output array
            }
        }
        
        return results
    }
    
    public func singleOutput(index: Int) -> Double?
    {
        //  Validate the index
        if (index < 0) { return nil}
        if (index >= size) { return nil }
        
        //  Get the data
        do {
            let outputs = try getOutput(index)
            return outputs[0]
        }
        catch {
            //  index error
            return nil
        }
    }
}

extension MLClassificationDataSet {
    
    public func getInputRange() -> [(minimum: Double, maximum: Double)]
    {
        //  Allocate the array of tuples
        var results : [(minimum: Double, maximum: Double)] = Array(count: inputDimension, repeatedValue: (minimum: Double.infinity, maximum: -Double.infinity))
        
        //  Go through each input
        for index in 0..<size {
            do {
                let input = try getInput(index)
                //  Go through each dimension
                for dimension in 0..<inputDimension {
                    if (input[dimension] < results[dimension].minimum) { results[dimension].minimum = input[dimension] }
                    if (input[dimension] > results[dimension].maximum) { results[dimension].maximum = input[dimension] }
                }
            }
            catch {
                //  Error getting input array
            }
        }
        
        return results
    }
    
    public func groupClasses() throws -> ClassificationData
    {
        if (dataType == .Regression)  { throw DataTypeError.InvalidDataType }
        
        //  If the data already has classification data, skip
        if (optionalData != nil) {
            if optionalData is ClassificationData { return optionalData as! ClassificationData }
        }
        
        //  Create a classification data addendum
        let classificationData = ClassificationData()
        
        //  Get the different data labels
        for index in 0..<size {
            let thisClass = try getClass(index)
            let thisClassIndex = classificationData.foundLabels.indexOf(thisClass)
            if let classIndex = thisClassIndex {
                //  Class label found, increment count
                classificationData.classCount[classIndex] += 1
                //  Add offset of data point
                classificationData.classOffsets[classIndex].append(index)
            }
            else {
                //  Class label not found, add it
                classificationData.foundLabels.append(thisClass)
                classificationData.classCount.append(1)     //  Start count at 1 - this instance
                classificationData.classOffsets.append([index]) //  First offset is this point
            }
        }
        
        //  Set the classification data as the optional data for the data set
        return classificationData
    }
    
    public func singleOutput(index: Int) -> Double?
    {
        //  Validate the index
        if (index < 0) { return nil}
        if (index >= size) { return nil }
        
        //  Get the data
        do {
            let outputClass = try getClass(index)
            return Double(outputClass)
        }
        catch {
            //  index error
            return nil
        }
    }
}


extension MLCombinedDataSet {
    
    public func singleOutput(index: Int) -> Double?
    {
        //  Validate the index
        if (index < 0) { return nil}
        if (index >= size) { return nil }
        
        //  Get the data
        do {
            switch dataType {
            case .Regression:
                let outputs = try getOutput(index)
                return outputs[0]
            case .Classification:
                let outputClass = try getClass(index)
                return Double(outputClass)
            case .RealAndClass:
                let outputClass = try getClass(index)       //  Class is a single output
                return Double(outputClass)
            }
        }
        catch {
            //  index error
            return nil
        }
    }
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
            let inputs = try testData.getInput(index)
            let result = try classifyOne(inputs)
            let expectedClass = try testData.getClass(index)
            if (result == expectedClass) {countCorrect += 1}
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
            let inputs = try testData.getInput(index)
            let results = try predictOne(inputs)
            
            //  Sum up the differences
            for nodeIndex in 0..<results.count {
                let outputs = try testData.getOutput(index)
                sum += abs(results[nodeIndex] - outputs[nodeIndex])
            }
            
        }
        
        return sum
    }
    
}

public class ClassificationData {
    var foundLabels: [Int] = []
    var classCount: [Int] = []
    var classOffsets: [[Int]] = []
    
    var numClasses: Int
    {
        return foundLabels.count
    }
}

