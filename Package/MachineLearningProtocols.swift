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
    case dataNotClassification
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


public enum DataSetType   //  data type
{
    case regression
    case classification
    case realAndClass           //  Has both real array and class (SVM predict values, etc.)
}

public protocol MLDataSet : class {     //  Machine learning data set provider
    var dataType : DataSetType { get }
    var inputDimension: Int  { get }
    var outputDimension: Int { get }
    var size: Int { get }
    var optionalData: AnyObject? { get set }      //  Optional data that can be temporarily added by methods using the data set
    
    func getInput(_ index: Int) throws ->[Double]
}

public protocol MLRegressionDataSet : MLDataSet {     //  Machine learning regression data set provider
    func getOutput(_ index: Int) throws ->[Double]
    
    func setOutput(_ index: Int, newOutput : [Double]) throws
}

public protocol MLClassificationDataSet : MLDataSet {     //  Machine learning classification data set provider
    func getClass(_ index: Int) throws ->Int
    
    func setClass(_ index: Int, newClass : Int) throws
}

public protocol MLCombinedDataSet : MLRegressionDataSet, MLClassificationDataSet {     //  Machine learning classification AND regression data set provider (for classes that can do both)
}

public protocol Classifier {
    func getInputDimension() -> Int
    func getParameterDimension() -> Int     //  May only be valid after training
    func getNumberOfClasses() -> Int        //  May only be valid after training
    func setParameters(_ parameters: [Double]) throws
    func setCustomInitializer(_ function: ((_ trainData: MLDataSet)->[Double])!)
    func getParameters() throws -> [Double]
    func trainClassifier(_ trainData: MLClassificationDataSet) throws
    func continueTrainingClassifier(_ trainData: MLClassificationDataSet) throws      //  Trains without initializing parameters first
    func classifyOne(_ inputs: [Double]) throws ->Int
    func classify(_ testData: MLClassificationDataSet) throws
}

public protocol Regressor {
    func getInputDimension() -> Int
    func getOutputDimension() -> Int
    func getParameterDimension() -> Int
    func setParameters(_ parameters: [Double]) throws
    func setCustomInitializer(_ function: ((_ trainData: MLDataSet)->[Double])!)
    func getParameters() throws -> [Double]
    func trainRegressor(_ trainData: MLRegressionDataSet) throws
    func continueTrainingRegressor(_ trainData: MLRegressionDataSet) throws      //  Trains without initializing parameters first
    func predictOne(_ inputs: [Double]) throws ->[Double]
    func predict(_ testData: MLRegressionDataSet) throws
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
#if os(Linux)
            let j = Int(random() % (size - i)) + i
#else
            let j = Int(arc4random_uniform(UInt32(size - i))) + i
#endif
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
        var results : [(minimum: Double, maximum: Double)] = Array(repeating: (minimum: Double.infinity, maximum: -Double.infinity), count: inputDimension)
        
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
        var results : [(minimum: Double, maximum: Double)] = Array(repeating: (minimum: Double.infinity, maximum: -Double.infinity), count: outputDimension)
        
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
    
    public func singleOutput(_ index: Int) -> Double?
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
        var results : [(minimum: Double, maximum: Double)] = Array(repeating: (minimum: Double.infinity, maximum: -Double.infinity), count: inputDimension)
        
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
        if (dataType == .regression)  { throw DataTypeError.invalidDataType }
        
        //  If the data already has classification data, skip
        if (optionalData != nil) {
            if optionalData is ClassificationData { return optionalData as! ClassificationData }
        }
        
        //  Create a classification data addendum
        let classificationData = ClassificationData()
        
        //  Get the different data labels
        for index in 0..<size {
            let thisClass = try getClass(index)
            let thisClassIndex = classificationData.foundLabels.index(of: thisClass)
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
    
    public func singleOutput(_ index: Int) -> Double?
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
    
    public func singleOutput(_ index: Int) -> Double?
    {
        //  Validate the index
        if (index < 0) { return nil}
        if (index >= size) { return nil }
        
        //  Get the data
        do {
            switch dataType {
            case .regression:
                let outputs = try getOutput(index)
                return outputs[0]
            case .classification:
                let outputClass = try getClass(index)
                return Double(outputClass)
            case .realAndClass:
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
    public func getClassificationPercentage(_ testData: DataSet) throws -> Double
    {
        //  Verify the data set is the right type
        if (testData.dataType != .classification) { throw DataTypeError.invalidDataType }
        
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
    public func getTotalAbsError(_ testData: DataSet) throws -> Double
    {
        //  Verify the data set is the right type
        if (testData.dataType != .regression) { throw DataTypeError.invalidDataType }
        
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

open class ClassificationData {
    var foundLabels: [Int] = []
    var classCount: [Int] = []
    var classOffsets: [[Int]] = []
    
    var numClasses: Int
    {
        return foundLabels.count
    }
}

