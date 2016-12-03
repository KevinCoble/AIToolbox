//
//  DataSet.swift
//  AIToolbox
//
//  Created by Kevin Coble on 12/6/15.
//  Copyright © 2015 Kevin Coble. All rights reserved.
//

import Foundation


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


open class DataSet : MLRegressionDataSet, MLClassificationDataSet, MLCombinedDataSet {
    open let dataType : DataSetType
    open let inputDimension: Int
    open let outputDimension: Int
    fileprivate var inputs: [[Double]]
    fileprivate var outputs: [[Double]]?
    fileprivate var classes: [Int]?
    open var optionalData: AnyObject?        //  Optional data that can be temporarily added by methods using the data set
    
    public init(dataType : DataSetType, inputDimension : Int, outputDimension : Int)
    {
        //  Remember the data parameters
        self.dataType = dataType
        self.inputDimension = inputDimension
        self.outputDimension = outputDimension
        
        //  Allocate data arrays
        inputs = []
        switch dataType {
        case .regression:
            outputs = []
        case .classification:
            classes = []
        case .realAndClass:
            outputs = []
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
    
    public init?(fromRegressionDataSet: MLRegressionDataSet)
    {
        //  Remember the data parameters
        self.dataType = .regression
        self.inputDimension = fromRegressionDataSet.inputDimension
        self.outputDimension = fromRegressionDataSet.outputDimension
        
        //  Copy data arrays
        inputs = []
        outputs = []
        classes = nil
        do {
            for index in 0..<fromRegressionDataSet.size {
                inputs.append(try fromRegressionDataSet.getInput(index))
                outputs!.append(try fromRegressionDataSet.getOutput(index))
            }
        }
        catch {
            return nil
        }
    }
    
    public init?(fromClassificationDataSet: MLClassificationDataSet)
    {
        //  Remember the data parameters
        self.dataType = .classification
        self.inputDimension = fromClassificationDataSet.inputDimension
        self.outputDimension = 1
        
        //  Copy data arrays
        inputs = []
        outputs = nil
        classes = []
        do {
            for index in 0..<fromClassificationDataSet.size {
                inputs.append(try fromClassificationDataSet.getInput(index))
                classes!.append(try fromClassificationDataSet.getClass(index))
            }
        }
        catch {
            return nil
        }
    }
    
    public init?(fromCombinedDataSet: MLCombinedDataSet)
    {
        //  Remember the data parameters
        self.dataType = .realAndClass
        self.inputDimension = fromCombinedDataSet.inputDimension
        self.outputDimension = fromCombinedDataSet.outputDimension
        
        //  Copy data arrays
        inputs = []
        outputs = []
        classes = []
        do {
            for index in 0..<fromCombinedDataSet.size {
                inputs.append(try fromCombinedDataSet.getInput(index))
                outputs!.append(try fromCombinedDataSet.getOutput(index))
                classes!.append(try fromCombinedDataSet.getClass(index))
            }
        }
        catch {
            return nil
        }
    }
    
    public init?(dataType : DataSetType, withInputsFrom: MLDataSet)
    {
        //  Remember the data parameters
        self.dataType = dataType
        self.inputDimension = withInputsFrom.inputDimension
        self.outputDimension = withInputsFrom.outputDimension
        
        //  Allocate data arrays
        inputs = []
        switch dataType {
        case .regression:
            outputs = []
        case .classification:
            classes = []
        case .realAndClass:
            outputs = []
            classes = []
        }
        
        //  Copy the inputs
        for index in 0..<withInputsFrom.size {
            do {
                inputs.append(try withInputsFrom.getInput(index))
            }
            catch {
                return nil
            }
        }
    }
   
    public init?(fromDataSet: MLDataSet, withEntries: [Int])
    {
        //  Remember the data parameters
        self.dataType = fromDataSet.dataType
        self.inputDimension = fromDataSet.inputDimension
        self.outputDimension = fromDataSet.outputDimension
        
        //  Allocate data arrays
        inputs = []
        switch dataType {
        case .regression:
            outputs = []
        case .classification:
            classes = []
        case .realAndClass:
            outputs = []
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
    
    public init?(fromDataSet: MLDataSet, withEntries: ArraySlice<Int>)
    {
        //  Remember the data parameters
        self.dataType = fromDataSet.dataType
        self.inputDimension = fromDataSet.inputDimension
        self.outputDimension = fromDataSet.outputDimension
        
        //  Allocate data arrays
        inputs = []
        switch dataType {
        case .regression:
            outputs = []
        case .classification:
            classes = []
        case .realAndClass:
            outputs = []
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
    open func includeEntries(fromDataSet: MLDataSet, withEntries: [Int]) throws
    {
        //  Make sure the dataset matches
        if dataType != fromDataSet.dataType { throw DataTypeError.invalidDataType }
        if inputDimension != fromDataSet.inputDimension { throw DataTypeError.wrongDimensionOnInput }
        if outputDimension != fromDataSet.outputDimension { throw DataTypeError.wrongDimensionOnOutput }
        
        //  Get cast versions of the input based on the type so we can get the output
        var regressionSet : MLRegressionDataSet?
        var classifierSet : MLClassificationDataSet?
        var combinedSet : MLCombinedDataSet?
        switch dataType {
        case .regression:
            regressionSet = fromDataSet as? MLRegressionDataSet
        case .classification:
            classifierSet = fromDataSet as? MLClassificationDataSet
        case .realAndClass:
            combinedSet = fromDataSet as? MLCombinedDataSet
        }
        
        //  Copy the entries
        for index in withEntries {
            if (index  < 0) { throw DataIndexError.negative }
            if (index  >= fromDataSet.size) { throw DataIndexError.indexAboveDataSetSize }
            inputs.append(try fromDataSet.getInput(index))
            switch dataType {
            case .regression:
                outputs!.append(try regressionSet!.getOutput(index))
            case .classification:
                classes!.append(try classifierSet!.getClass(index))
            case .realAndClass:
                outputs!.append(try combinedSet!.getOutput(index))
                classes!.append(try combinedSet!.getClass(index))
            }
        }
    }
    
    ///  Get entries from another matching dataset
    open func includeEntries(fromDataSet: MLDataSet, withEntries: ArraySlice<Int>) throws
    {
        //  Make sure the dataset matches
        if dataType != fromDataSet.dataType { throw DataTypeError.invalidDataType }
        if inputDimension != fromDataSet.inputDimension { throw DataTypeError.wrongDimensionOnInput }
        if outputDimension != fromDataSet.outputDimension { throw DataTypeError.wrongDimensionOnOutput }
        
        //  Get cast versions of the input based on the type so we can get the output
        var regressionSet : MLRegressionDataSet?
        var classifierSet : MLClassificationDataSet?
        var combinedSet : MLCombinedDataSet?
        switch dataType {
        case .regression:
            regressionSet = fromDataSet as? MLRegressionDataSet
        case .classification:
            classifierSet = fromDataSet as? MLClassificationDataSet
        case .realAndClass:
            combinedSet = fromDataSet as? MLCombinedDataSet
        }
        
        //  Copy the entries
        for index in withEntries {
            if (index  < 0) { throw DataIndexError.negative }
            if (index  >= fromDataSet.size) { throw DataIndexError.indexAboveDataSetSize }
            inputs.append(try fromDataSet.getInput(index))
            switch dataType {
            case .regression:
                outputs!.append(try regressionSet!.getOutput(index))
            case .classification:
                classes!.append(try classifierSet!.getClass(index))
            case .realAndClass:
                outputs!.append(try combinedSet!.getOutput(index))
                classes!.append(try combinedSet!.getClass(index))
            }
        }
    }
    
    ///  Get inputs from another matching dataset, setting outputs to 0
    open func includeEntryInputs(fromDataSet: MLDataSet, withEntries: [Int]) throws
    {
        //  Make sure the dataset inputs match
        if inputDimension != fromDataSet.inputDimension { throw DataTypeError.wrongDimensionOnInput }
        
        //  Copy the inputs
        for index in withEntries {
            if (index  < 0) { throw DataIndexError.negative }
            if (index  >= fromDataSet.size) { throw DataIndexError.indexAboveDataSetSize }
            inputs.append(try fromDataSet.getInput(index))
            switch dataType {
            case .regression:
                outputs!.append([Double](repeating: 0.0, count: outputDimension))
            case .classification:
                classes!.append(0)
            case .realAndClass:
                outputs!.append([Double](repeating: 0.0, count: outputDimension))
                classes!.append(0)
            }
        }
    }
    
    ///  Get inputs from another matching dataset, setting outputs to 0
    open func includeEntryInputs(fromDataSet: MLDataSet, withEntries: ArraySlice<Int>) throws
    {
        //  Make sure the dataset inputs match
        if inputDimension != fromDataSet.inputDimension { throw DataTypeError.wrongDimensionOnInput }
        
        //  Copy the inputs
        for index in withEntries {
            if (index  < 0) { throw DataIndexError.negative }
            if (index  >= fromDataSet.size) { throw DataIndexError.indexAboveDataSetSize }
            inputs.append(try fromDataSet.getInput(index))
            switch dataType {
            case .regression:
                outputs!.append([Double](repeating: 0.0, count: outputDimension))
            case .classification:
                classes!.append(0)
            case .realAndClass:
                outputs!.append([Double](repeating: 0.0, count: outputDimension))
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
        if (dataType == .classification) { throw DataTypeError.dataWrongForType }
        if (input.count != inputDimension) { throw DataTypeError.wrongDimensionOnInput }
        if (output.count != outputDimension) { throw DataTypeError.wrongDimensionOnOutput }
        
        //  Add the new data item
        inputs.append(input)
        outputs!.append(output)
        if (dataType == .realAndClass) { classes!.append(0) }
    }
    
    open func addDataPoint(input : [Double], dataClass: Int) throws
    {
        //  Validate the data
        if (dataType == .regression) { throw DataTypeError.dataWrongForType }
        if (input.count != inputDimension) { throw DataTypeError.wrongDimensionOnInput }
        
        //  Add the new data item
        inputs.append(input)
        classes!.append(dataClass)
        if (dataType == .realAndClass) { outputs!.append([Double](repeating: 0.0, count: outputDimension)) }
    }
    
    open func addDataPoint(input : [Double], output: [Double], dataClass: Int) throws
    {
        //  Validate the data
        if (dataType != .realAndClass) { throw DataTypeError.dataWrongForType }
        if (input.count != inputDimension) { throw DataTypeError.wrongDimensionOnInput }
        if (output.count != outputDimension) { throw DataTypeError.wrongDimensionOnOutput }
        
        //  Add the new data item
        inputs.append(input)
        outputs!.append(output)
        classes!.append(dataClass)
    }
    
    open func setOutput(_ index: Int, newOutput : [Double]) throws
    {
        //  Validate the data
        if (dataType == .classification) { throw DataTypeError.dataWrongForType }
        if (index < 0) { throw  DataIndexError.negative }
        if (index > inputs.count) { throw  DataIndexError.indexAboveDimension }
        if (newOutput.count != outputDimension) { throw DataTypeError.wrongDimensionOnOutput }
        
        //  Make sure we have outputs up until this index (we have the inputs already)
        if (index >= outputs!.count) {
            while (index > outputs!.count) {    //  Insert any uncreated data between this index and existing values
                outputs!.append([Double](repeating: 0.0, count: outputDimension))
                if (dataType == .realAndClass) { classes!.append(0) }
            }
            //  Append the new data
            outputs!.append(newOutput)
            if (dataType == .realAndClass) { classes!.append(0) }
        }
        
        else {
            //  Replace the new output item
            outputs![index] = newOutput
        }
    }
    
    open func setClass(_ index: Int, newClass : Int) throws
    {
        //  Validate the data
        if (dataType == .regression) { throw DataTypeError.dataWrongForType }
        if (index < 0) { throw  DataIndexError.negative }
        if (index > inputs.count) { throw  DataIndexError.negative }
        
        //  Make sure we have class labels up until this index (we have the inputs already)
        if (index >= classes!.count) {
            while (index > classes!.count) {    //  Insert any uncreated data between this index and existing values
                classes!.append(0)
                if (dataType == .realAndClass) { outputs!.append([Double](repeating: 0.0, count: outputDimension)) }
            }
            //  Append the new data
            classes!.append(newClass)
            if (dataType == .realAndClass) { outputs!.append([Double](repeating: 0.0, count: outputDimension)) }
        }
            
        else {
            //  Replace the new output item
            classes![index] = newClass
        }
        
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
        if (dataType == .classification) { throw DataTypeError.dataWrongForType }
        if (index < 0) { throw  DataIndexError.negative }
        if (index > outputs!.count) { throw  DataIndexError.indexAboveDataSetSize }
        
        return outputs![index]
    }
   
    open func getClass(_ index: Int) throws ->Int
    {
        //  Validate the data
        if (dataType == .regression) { throw DataTypeError.dataWrongForType }
        if (index < 0) { throw  DataIndexError.negative }
        if (index > classes!.count) { throw  DataIndexError.indexAboveDataSetSize }
        
        return classes![index]
    }
    
    //  Leave here in case it is used by other methods
    open static func gaussianRandom(_ mean : Double, standardDeviation : Double) -> Double
    {
        return Gaussian.gaussianRandom(mean, standardDeviation: standardDeviation)
    }
}
