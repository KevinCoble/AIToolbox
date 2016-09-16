//
//  DataSet.swift
//  AIToolbox
//
//  Created by Kevin Coble on 12/6/15.
//  Copyright © 2015 Kevin Coble. All rights reserved.
//

import Foundation


public enum DataSetType   //  data type
{
    case regression
    case classification
}

enum DataTypeError: Error {
    case invalidDataType
    case dataWrongForType
    case wrongDimensionOnInput
    case wrongDimensionOnOutput
}

enum DataIndexError: Error {
    case negative
    case indexAboveDimension
    case indexAboveDataSetSize
}


open class DataSet {
    let dataType : DataSetType
    let inputDimension: Int
    let outputDimension: Int
    var inputs: [[Double]]
    var outputs: [[Double]]?
    var classes: [Int]?
    var optionalData: AnyObject?        //  Optional data that can be temporarily added by methods using the data
    
    public init(dataType : DataSetType, inputDimension : Int, outputDimension : Int)
    {
        //  Remember the data parameters
        self.dataType = dataType
        self.inputDimension = inputDimension
        self.outputDimension = outputDimension
        
        //  Allocate data arrays
        inputs = []
        if (dataType == .regression) {
            outputs = []
        }
        else {
            classes = []
        }
    }
    
    public init(fromDataSet: DataSet)
    {
        //  Remember the data parameters
        self.dataType = fromDataSet.dataType
        self.inputDimension = fromDataSet.inputDimension
        self.outputDimension = fromDataSet.outputDimension
        
        //  Copy data arrays
        inputs = fromDataSet.inputs
        outputs = fromDataSet.outputs
        classes = fromDataSet.classes
    }
   
    public init?(fromDataSet: DataSet, withEntries: [Int])
    {
        //  Remember the data parameters
        self.dataType = fromDataSet.dataType
        self.inputDimension = fromDataSet.inputDimension
        self.outputDimension = fromDataSet.outputDimension
        
        //  Allocate data arrays
        inputs = []
        if (dataType == .regression) {
            outputs = []
        }
        else {
            classes = []
        }
        
        //  Copy the entries
        do {
            try includeEntries(fromDataSet: fromDataSet, withEntries: withEntries)
        }
        catch {
            return nil
        }
    }
    
    public init?(fromDataSet: DataSet, withEntries: ArraySlice<Int>)
    {
        //  Remember the data parameters
        self.dataType = fromDataSet.dataType
        self.inputDimension = fromDataSet.inputDimension
        self.outputDimension = fromDataSet.outputDimension
        
        //  Allocate data arrays
        inputs = []
        if (dataType == .regression) {
            outputs = []
        }
        else {
            classes = []
        }
        
        //  Copy the entries
        do {
            try includeEntries(fromDataSet: fromDataSet, withEntries: withEntries)
        }
        catch {
            return nil
        }
    }
    
    ///  Get entries from another matching dataset
    open func includeEntries(fromDataSet: DataSet, withEntries: [Int]) throws
    {
        //  Make sure the dataset matches
        if dataType != fromDataSet.dataType { throw DataTypeError.invalidDataType }
        if inputDimension != fromDataSet.inputDimension { throw DataTypeError.wrongDimensionOnInput }
        if outputDimension != fromDataSet.outputDimension { throw DataTypeError.wrongDimensionOnOutput }
        
        //  Copy the entries
        for index in withEntries {
            if (index  < 0) { throw DataIndexError.negative }
            if (index  >= fromDataSet.size) { throw DataIndexError.indexAboveDataSetSize }
            inputs.append(fromDataSet.inputs[index])
            if (dataType == .regression) {
                outputs!.append(fromDataSet.outputs![index])
            }
            else {
                classes!.append(fromDataSet.classes![index])
                if outputs != nil {
                    outputs!.append(fromDataSet.outputs![index])
                }
            }
        }
    }
    
    ///  Get entries from another matching dataset
    open func includeEntries(fromDataSet: DataSet, withEntries: ArraySlice<Int>) throws
    {
        //  Make sure the dataset matches
        if dataType != fromDataSet.dataType { throw DataTypeError.invalidDataType }
        if inputDimension != fromDataSet.inputDimension { throw DataTypeError.wrongDimensionOnInput }
        if outputDimension != fromDataSet.outputDimension { throw DataTypeError.wrongDimensionOnOutput }
        
        //  Copy the entries
        for index in withEntries {
            if (index  < 0) { throw DataIndexError.negative }
            if (index  >= fromDataSet.size) { throw DataIndexError.indexAboveDataSetSize }
            inputs.append(fromDataSet.inputs[index])
            if (dataType == .regression) {
                outputs!.append(fromDataSet.outputs![index])
            }
            else {
                classes!.append(fromDataSet.classes![index])
                if outputs != nil {
                    outputs!.append(fromDataSet.outputs![index])
                }
            }
        }
    }
    
    ///  Get inputs from another matching dataset, initializing outputs to 0
    open func includeEntryInputs(fromDataSet: DataSet, withEntries: [Int]) throws
    {
        //  Make sure the dataset inputs match
        if inputDimension != fromDataSet.inputDimension { throw DataTypeError.wrongDimensionOnInput }
        
        //  Copy the inputs
        for index in withEntries {
            if (index  < 0) { throw DataIndexError.negative }
            if (index  >= fromDataSet.size) { throw DataIndexError.indexAboveDataSetSize }
            inputs.append(fromDataSet.inputs[index])
            if (dataType == .regression) {
                outputs!.append([Double](repeating: 0.0, count: outputDimension))
            }
            else {
                classes!.append(0)
            }
        }
    }
    
    ///  Get inputs from another matching dataset, initializing outputs to 0
    open func includeEntryInputs(fromDataSet: DataSet, withEntries: ArraySlice<Int>) throws
    {
        //  Make sure the dataset inputs match
        if inputDimension != fromDataSet.inputDimension { throw DataTypeError.wrongDimensionOnInput }
        
        //  Copy the inputs
        for index in withEntries {
            if (index  < 0) { throw DataIndexError.negative }
            if (index  >= fromDataSet.size) { throw DataIndexError.indexAboveDataSetSize }
            inputs.append(fromDataSet.inputs[index])
            if (dataType == .regression) {
                outputs!.append([Double](repeating: 0.0, count: outputDimension))
            }
            else {
                classes!.append(0)
            }
        }
    }
    
    open var size: Int
    {
        return inputs.count
    }
    
    open func singleOutput(_ index: Int) -> Double?
    {
        //  Validate the index
        if (index < 0) { return nil}
        if (index >= inputs.count) { return nil }
        
        //  Get the data
        if (dataType == .regression) {
            return outputs![index][0]
        }
        else {
            return Double(classes![index])
        }
    }
    
    open func addDataPoint(input : [Double], output: [Double]) throws
    {
        //  Validate the data
        if (dataType != .regression) { throw DataTypeError.dataWrongForType }
        if (input.count != inputDimension) { throw DataTypeError.wrongDimensionOnInput }
        if (output.count != outputDimension) { throw DataTypeError.wrongDimensionOnOutput }
        
        //  Add the new data item
        inputs.append(input)
        outputs!.append(output)
    }
    
    open func addDataPoint(input : [Double], output: Int) throws
    {
        //  Validate the data
        if (dataType != .classification) { throw DataTypeError.dataWrongForType }
        if (input.count != inputDimension) { throw DataTypeError.wrongDimensionOnInput }
        
        //  Add the new data item
        inputs.append(input)
        classes!.append(output)
    }
    
    open func setOutput(_ index: Int, newOutput : [Double]) throws
    {
        //  Validate the data
        if (dataType != .regression) { throw DataTypeError.dataWrongForType }
        if (index < 0) { throw  DataIndexError.negative }
        if (index > inputs.count) { throw  DataIndexError.negative }
        if (newOutput.count != outputDimension) { throw DataTypeError.wrongDimensionOnOutput }
        
        //  Add the new output item
        outputs![index] = newOutput
    }
    
    open func setClass(_ index: Int, newClass : Int) throws
    {
        //  Validate the data
        if (dataType != .classification) { throw DataTypeError.dataWrongForType }
        if (index < 0) { throw  DataIndexError.negative }
        if (index > inputs.count) { throw  DataIndexError.negative }
        
        classes![index] = newClass
    }
    
    open func addUnlabeledDataPoint(input : [Double]) throws
    {
        //  Validate the data
        if (input.count != inputDimension) { throw DataTypeError.wrongDimensionOnInput }
        
        //  Add the new data item
        inputs.append(input)
    }
    open func addTestDataPoint(input : [Double]) throws
    {
        //  Validate the data
        if (input.count != inputDimension) { throw DataTypeError.wrongDimensionOnInput }
        
        //  Add the new data item
        inputs.append(input)
    }
    
    open func getInput(_ index: Int) throws ->[Double]
    {
        //  Validate the data
        if (index < 0) { throw  DataIndexError.negative }
        if (index > inputs.count) { throw  DataIndexError.indexAboveDataSetSize }
        
        return inputs[index]
    }
    
    open func getOutput(_ index: Int) throws ->[Double]
    {
        //  Validate the data
        if (dataType != .regression) { throw DataTypeError.dataWrongForType }
        if (index < 0) { throw  DataIndexError.negative }
        if (index > outputs!.count) { throw  DataIndexError.indexAboveDataSetSize }
        
        return outputs![index]
    }
   
    open func getClass(_ index: Int) throws ->Int
    {
        //  Validate the data
        if (dataType != .classification) { throw DataTypeError.dataWrongForType }
        if (index < 0) { throw  DataIndexError.negative }
        if (index > classes!.count) { throw  DataIndexError.indexAboveDataSetSize }
        
        return classes![index]
    }
    
    open func getRandomIndexSet() -> [Int]
    {
        //  Get the ordered array of indices
        var shuffledArray: [Int] = []
        for i in 0..<inputs.count { shuffledArray.append(i) }
        
        // empty and single-element collections don't shuffle
        if size < 2 { return shuffledArray }
        
        //  Shuffle
        for i in 0..<inputs.count - 1 {
            let j = Int(arc4random_uniform(UInt32(inputs.count - i))) + i
            guard i != j else { continue }
            swap(&shuffledArray[i], &shuffledArray[j])
        }
        
        return shuffledArray
    }
    
    open func getInputRange() -> [(minimum: Double, maximum: Double)]
    {
        //  Allocate the array of tuples
        var results : [(minimum: Double, maximum: Double)] = Array(repeating: (minimum: Double.infinity, maximum: -Double.infinity), count: inputDimension)
        
        //  Go through each input
        for input in inputs {
            //  Go through each dimension
            for dimension in 0..<inputDimension {
                if (input[dimension] < results[dimension].minimum) { results[dimension].minimum = input[dimension] }
                if (input[dimension] > results[dimension].maximum) { results[dimension].maximum = input[dimension] }
            }
        }
        
        return results
    }
    
    open func getOutputRange() -> [(minimum: Double, maximum: Double)]
    {
        //  Allocate the array of tuples
        var results : [(minimum: Double, maximum: Double)] = Array(repeating: (minimum: Double.infinity, maximum: -Double.infinity), count: outputDimension)
        
        //  If no outputs, return invalid range
        if (outputs == nil) { return results }
        
        //  Go through each output
        for output in outputs! {
            //  Go through each dimension
            for dimension in 0..<outputDimension {
                if (output[dimension] < results[dimension].minimum) { results[dimension].minimum = output[dimension] }
                if (output[dimension] > results[dimension].maximum) { results[dimension].maximum = output[dimension] }
            }
        }
        
        return results
    }
    
    open func groupClasses() throws
    {
        if (dataType != .classification)  { throw DataTypeError.invalidDataType }
        
        //  If the data already has classification data, skip
        if (optionalData != nil) {
            if optionalData is ClassificationData { return }
        }
        
        //  Create a classification data addendum
        let classificationData = ClassificationData()
        
        //  Get the different data labels
        for index in 0..<size {
            let thisClass = classes![index]
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
        optionalData = classificationData
    }

    
    //  Leave here in case it is used by other methods
    open static func gaussianRandom(_ mean : Double, standardDeviation : Double) -> Double
    {
        return Gaussian.gaussianRandom(mean, standardDeviation: standardDeviation)
    }
}
