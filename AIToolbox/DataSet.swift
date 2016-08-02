//
//  DataSet.swift
//  AIToolbox
//
//  Created by Kevin Coble on 12/6/15.
//  Copyright © 2015 Kevin Coble. All rights reserved.
//

import Foundation


enum DataTypeError: ErrorType {
    case InvalidDataType
    case DataWrongForType
    case WrongDimensionOnInput
    case WrongDimensionOnOutput
}

enum DataIndexError: ErrorType {
    case Negative
    case IndexAboveDimension
    case IndexAboveDataSetSize
}


public class DataSet : MLRegressionDataSet, MLClassificationDataSet, MLCombinedDataSet {
    public let dataType : DataSetType
    public let inputDimension: Int
    public let outputDimension: Int
    private var inputs: [[Double]]
    private var outputs: [[Double]]?
    private var classes: [Int]?
    public var optionalData: AnyObject?        //  Optional data that can be temporarily added by methods using the data set
    
    public init(dataType : DataSetType, inputDimension : Int, outputDimension : Int)
    {
        //  Remember the data parameters
        self.dataType = dataType
        self.inputDimension = inputDimension
        self.outputDimension = outputDimension
        
        //  Allocate data arrays
        inputs = []
        switch dataType {
        case .Regression:
            outputs = []
        case .Classification:
            classes = []
        case .RealAndClass:
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
        self.dataType = .Regression
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
        self.dataType = .Classification
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
        self.dataType = .RealAndClass
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
        case .Regression:
            outputs = []
        case .Classification:
            classes = []
        case .RealAndClass:
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
        case .Regression:
            outputs = []
        case .Classification:
            classes = []
        case .RealAndClass:
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
        case .Regression:
            outputs = []
        case .Classification:
            classes = []
        case .RealAndClass:
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
    public func includeEntries(fromDataSet fromDataSet: MLDataSet, withEntries: [Int]) throws
    {
        //  Make sure the dataset matches
        if dataType != fromDataSet.dataType { throw DataTypeError.InvalidDataType }
        if inputDimension != fromDataSet.inputDimension { throw DataTypeError.WrongDimensionOnInput }
        if outputDimension != fromDataSet.outputDimension { throw DataTypeError.WrongDimensionOnOutput }
        
        //  Get cast versions of the input based on the type so we can get the output
        var regressionSet : MLRegressionDataSet?
        var classifierSet : MLClassificationDataSet?
        var combinedSet : MLCombinedDataSet?
        switch dataType {
        case .Regression:
            regressionSet = fromDataSet as? MLRegressionDataSet
        case .Classification:
            classifierSet = fromDataSet as? MLClassificationDataSet
        case .RealAndClass:
            combinedSet = fromDataSet as? MLCombinedDataSet
        }
        
        //  Copy the entries
        for index in withEntries {
            if (index  < 0) { throw DataIndexError.Negative }
            if (index  >= fromDataSet.size) { throw DataIndexError.IndexAboveDataSetSize }
            inputs.append(try fromDataSet.getInput(index))
            switch dataType {
            case .Regression:
                outputs!.append(try regressionSet!.getOutput(index))
            case .Classification:
                classes!.append(try classifierSet!.getClass(index))
            case .RealAndClass:
                outputs!.append(try combinedSet!.getOutput(index))
                classes!.append(try combinedSet!.getClass(index))
            }
        }
    }
    
    ///  Get entries from another matching dataset
    public func includeEntries(fromDataSet fromDataSet: MLDataSet, withEntries: ArraySlice<Int>) throws
    {
        //  Make sure the dataset matches
        if dataType != fromDataSet.dataType { throw DataTypeError.InvalidDataType }
        if inputDimension != fromDataSet.inputDimension { throw DataTypeError.WrongDimensionOnInput }
        if outputDimension != fromDataSet.outputDimension { throw DataTypeError.WrongDimensionOnOutput }
        
        //  Get cast versions of the input based on the type so we can get the output
        var regressionSet : MLRegressionDataSet?
        var classifierSet : MLClassificationDataSet?
        var combinedSet : MLCombinedDataSet?
        switch dataType {
        case .Regression:
            regressionSet = fromDataSet as? MLRegressionDataSet
        case .Classification:
            classifierSet = fromDataSet as? MLClassificationDataSet
        case .RealAndClass:
            combinedSet = fromDataSet as? MLCombinedDataSet
        }
        
        //  Copy the entries
        for index in withEntries {
            if (index  < 0) { throw DataIndexError.Negative }
            if (index  >= fromDataSet.size) { throw DataIndexError.IndexAboveDataSetSize }
            inputs.append(try fromDataSet.getInput(index))
            switch dataType {
            case .Regression:
                outputs!.append(try regressionSet!.getOutput(index))
            case .Classification:
                classes!.append(try classifierSet!.getClass(index))
            case .RealAndClass:
                outputs!.append(try combinedSet!.getOutput(index))
                classes!.append(try combinedSet!.getClass(index))
            }
        }
    }
    
    ///  Get inputs from another matching dataset, setting outputs to 0
    public func includeEntryInputs(fromDataSet fromDataSet: MLDataSet, withEntries: [Int]) throws
    {
        //  Make sure the dataset inputs match
        if inputDimension != fromDataSet.inputDimension { throw DataTypeError.WrongDimensionOnInput }
        
        //  Copy the inputs
        for index in withEntries {
            if (index  < 0) { throw DataIndexError.Negative }
            if (index  >= fromDataSet.size) { throw DataIndexError.IndexAboveDataSetSize }
            inputs.append(try fromDataSet.getInput(index))
            switch dataType {
            case .Regression:
                outputs!.append([Double](count:outputDimension, repeatedValue: 0.0))
            case .Classification:
                classes!.append(0)
            case .RealAndClass:
                outputs!.append([Double](count:outputDimension, repeatedValue: 0.0))
                classes!.append(0)
            }
        }
    }
    
    ///  Get inputs from another matching dataset, setting outputs to 0
    public func includeEntryInputs(fromDataSet fromDataSet: MLDataSet, withEntries: ArraySlice<Int>) throws
    {
        //  Make sure the dataset inputs match
        if inputDimension != fromDataSet.inputDimension { throw DataTypeError.WrongDimensionOnInput }
        
        //  Copy the inputs
        for index in withEntries {
            if (index  < 0) { throw DataIndexError.Negative }
            if (index  >= fromDataSet.size) { throw DataIndexError.IndexAboveDataSetSize }
            inputs.append(try fromDataSet.getInput(index))
            switch dataType {
            case .Regression:
                outputs!.append([Double](count:outputDimension, repeatedValue: 0.0))
            case .Classification:
                classes!.append(0)
            case .RealAndClass:
                outputs!.append([Double](count:outputDimension, repeatedValue: 0.0))
                classes!.append(0)
            }
        }
    }
    
    public var size: Int
    {
        return inputs.count
    }
    
    public func singleOutput(index: Int) -> Double?
    {
        //  Validate the index
        if (index < 0) { return nil}
        if (index >= inputs.count) { return nil }
        
        //  Get the data
        if (dataType == .Regression) {
            return outputs![index][0]
        }
        else {
            return Double(classes![index])
        }
    }
    
    public func addDataPoint(input input : [Double], output: [Double]) throws
    {
        //  Validate the data
        if (dataType == .Classification) { throw DataTypeError.DataWrongForType }
        if (input.count != inputDimension) { throw DataTypeError.WrongDimensionOnInput }
        if (output.count != outputDimension) { throw DataTypeError.WrongDimensionOnOutput }
        
        //  Add the new data item
        inputs.append(input)
        outputs!.append(output)
        if (dataType == .RealAndClass) { classes!.append(0) }
    }
    
    public func addDataPoint(input input : [Double], dataClass: Int) throws
    {
        //  Validate the data
        if (dataType == .Regression) { throw DataTypeError.DataWrongForType }
        if (input.count != inputDimension) { throw DataTypeError.WrongDimensionOnInput }
        
        //  Add the new data item
        inputs.append(input)
        classes!.append(dataClass)
        if (dataType == .RealAndClass) { outputs!.append([Double](count:outputDimension, repeatedValue: 0.0)) }
    }
    
    public func addDataPoint(input input : [Double], output: [Double], dataClass: Int) throws
    {
        //  Validate the data
        if (dataType != .RealAndClass) { throw DataTypeError.DataWrongForType }
        if (input.count != inputDimension) { throw DataTypeError.WrongDimensionOnInput }
        if (output.count != outputDimension) { throw DataTypeError.WrongDimensionOnOutput }
        
        //  Add the new data item
        inputs.append(input)
        outputs!.append(output)
        classes!.append(dataClass)
    }
    
    public func setOutput(index: Int, newOutput : [Double]) throws
    {
        //  Validate the data
        if (dataType == .Classification) { throw DataTypeError.DataWrongForType }
        if (index < 0) { throw  DataIndexError.Negative }
        if (index > inputs.count) { throw  DataIndexError.IndexAboveDimension }
        if (newOutput.count != outputDimension) { throw DataTypeError.WrongDimensionOnOutput }
        
        //  Make sure we have outputs up until this index (we have the inputs already)
        if (index >= outputs!.count) {
            while (index > outputs!.count) {    //  Insert any uncreated data between this index and existing values
                outputs!.append([Double](count:outputDimension, repeatedValue: 0.0))
                if (dataType == .RealAndClass) { classes!.append(0) }
            }
            //  Append the new data
            outputs!.append(newOutput)
            if (dataType == .RealAndClass) { classes!.append(0) }
        }
        
        else {
            //  Replace the new output item
            outputs![index] = newOutput
        }
    }
    
    public func setClass(index: Int, newClass : Int) throws
    {
        //  Validate the data
        if (dataType == .Regression) { throw DataTypeError.DataWrongForType }
        if (index < 0) { throw  DataIndexError.Negative }
        if (index > inputs.count) { throw  DataIndexError.Negative }
        
        //  Make sure we have class labels up until this index (we have the inputs already)
        if (index >= classes!.count) {
            while (index > classes!.count) {    //  Insert any uncreated data between this index and existing values
                classes!.append(0)
                if (dataType == .RealAndClass) { outputs!.append([Double](count:outputDimension, repeatedValue: 0.0)) }
            }
            //  Append the new data
            classes!.append(newClass)
            if (dataType == .RealAndClass) { outputs!.append([Double](count:outputDimension, repeatedValue: 0.0)) }
        }
            
        else {
            //  Replace the new output item
            classes![index] = newClass
        }
        
        classes![index] = newClass
    }
    
    public func addUnlabeledDataPoint(input input : [Double]) throws
    {
        //  Validate the data
        if (input.count != inputDimension) { throw DataTypeError.WrongDimensionOnInput }
        
        //  Add the new data item
        inputs.append(input)
    }
    public func addTestDataPoint(input input : [Double]) throws
    {
        //  Validate the data
        if (input.count != inputDimension) { throw DataTypeError.WrongDimensionOnInput }
        
        //  Add the new data item
        inputs.append(input)
    }
    
    public func getInput(index: Int) throws ->[Double]
    {
        //  Validate the data
        if (index < 0) { throw  DataIndexError.Negative }
        if (index > inputs.count) { throw  DataIndexError.IndexAboveDataSetSize }
        
        return inputs[index]
    }
    
    public func getOutput(index: Int) throws ->[Double]
    {
        //  Validate the data
        if (dataType == .Classification) { throw DataTypeError.DataWrongForType }
        if (index < 0) { throw  DataIndexError.Negative }
        if (index > outputs!.count) { throw  DataIndexError.IndexAboveDataSetSize }
        
        return outputs![index]
    }
   
    public func getClass(index: Int) throws ->Int
    {
        //  Validate the data
        if (dataType == .Regression) { throw DataTypeError.DataWrongForType }
        if (index < 0) { throw  DataIndexError.Negative }
        if (index > classes!.count) { throw  DataIndexError.IndexAboveDataSetSize }
        
        return classes![index]
    }
    
    //  Leave here in case it is used by other methods
    public static func gaussianRandom(mean : Double, standardDeviation : Double) -> Double
    {
        return Gaussian.gaussianRandom(mean, standardDeviation: standardDeviation)
    }
}
