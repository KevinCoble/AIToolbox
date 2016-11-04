//
//  DeepChannel.swift
//  AIToolbox
//
//  Created by Kevin Coble on 6/25/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation

public struct DeepChannelSize {
    public let numDimensions : Int
    public var dimensions : [Int]
    
    
    public init(dimensionCount: Int, dimensionValues: [Int]) {
        numDimensions = dimensionCount
        dimensions = dimensionValues
    }
    
    public var totalSize: Int {
        get {
            var result = 1
            for i in 0..<numDimensions {
                result *= dimensions[i]
            }
            return result
        }
    }
    
    public func asString() ->String
    {
        var result = "\(numDimensions)D - ["
        if (numDimensions > 0) { result += "\(dimensions[0])" }
        if (numDimensions > 1) {
            for i in 1..<numDimensions {
                result += ", \(dimensions[i])"
            }
        }
        result += "]"
        
        return result
    }
}

///  Class for a single channel of a deep layer
///  A deep channel manages a network topology for a single data-stream within a deep layer
///  It contains an ordered array of 'network operators' that manipulate the channel data (convolutions, poolings, feedforward nets, etc.)
final public class DeepChannel : MLPersistence
{
    public let idString : String           //  The string ID for the channel.  i.e. "red component"
    public private(set) var sourceChannelID : String    //  ID of the channel that is the source for this channel from the previous layer
    public private(set) var resultSize : DeepChannelSize    //  Size of the result of this channel
    
    var networkOperators : [DeepNetworkOperator] = []
    
    fileprivate var inputErrorGradient : [Float] = []
    
    public init(identifier: String, sourceChannel: String) {
        idString = identifier
        sourceChannelID = sourceChannel
        resultSize = DeepChannelSize(dimensionCount: 1, dimensionValues: [0])
    }
    
    public init?(fromDictionary: [String: AnyObject])
    {
        //  Init for nil return (hopefully Swift 3 removes this need)
        resultSize = DeepChannelSize(dimensionCount: 1, dimensionValues: [0])
        
        //  Get the id string type
        let id = fromDictionary["idString"] as? NSString
        if id == nil { return nil }
        idString = id! as String
        
        //  Get the source ID
        let source = fromDictionary["sourceChannelID"] as? NSString
        if source == nil { return nil }
        sourceChannelID = source! as String
        
        //  Get the array of network operators
        let networkOpArray = fromDictionary["networkOperators"] as? NSArray
        if (networkOpArray == nil)  { return nil }
        for item in networkOpArray! {
            let element = item as? [String: AnyObject]
            if (element == nil)  { return nil }
            let netOperator = DeepNetworkOperatorType.getDeepNetworkOperatorFromDict(element!)
            if (netOperator == nil)  { return nil }
            networkOperators.append(netOperator!)
        }
    }
    
    ///  Function to add a network operator to the channel
    public func addNetworkOperator(_ newOperator: DeepNetworkOperator)
    {
        networkOperators.append(newOperator)
    }
    
    
    ///  Function to get the number of defined operators
    public var numOperators: Int {
        get { return networkOperators.count }
    }
    
    ///  Function to get a network operator at the specified index
    public func getNetworkOperator(_ operatorIndex: Int) ->DeepNetworkOperator?
    {
        if (operatorIndex >= 0 && operatorIndex < networkOperators.count) {
           return networkOperators[operatorIndex]
        }
        
        return nil
    }
    
    
    ///  Function to replace a network operator at the specified index
    public func replaceNetworkOperator(_ operatorIndex: Int, newOperator: DeepNetworkOperator)
    {
        if (operatorIndex >= 0 && operatorIndex < networkOperators.count) {
            networkOperators[operatorIndex] = newOperator
        }
    }
    
    
    ///  Functions to remove a network operator from the channel
    public func removeNetworkOperator(_ operatorIndex: Int)
    {
        if (operatorIndex >= 0 && operatorIndex < networkOperators.count) {
            networkOperators.remove(at: operatorIndex)
        }
    }
    
    //  Method to determine the output size based on the input size and the operation layers
    func updateOutputSize(_ inputSize : DeepChannelSize)
    {
        //  Iterate through each operator, adjusting the size
        var currentSize = inputSize
        for networkOperator in networkOperators {
            currentSize = networkOperator.getResultingSize(currentSize)
        }
        resultSize = currentSize
    }
    
    
    func getResultRange() ->(minimum: Float, maximum: Float)
    {
        if let lastOperator = networkOperators.last {
            return lastOperator.getResultRange()
        }
        return (minimum: 0.0, maximum: 1.0)
    }
    
    public func initializeParameters()
    {
        for networkOperator in networkOperators {
            networkOperator.initializeParameters()
        }
    }

    
    //  Method to feed values forward through the channel
    func feedForward(_ inputSource: DeepNetworkInputSource)
    {
        //  Get the inputs from the previous layer
        var inputs = inputSource.getValuesForID(sourceChannelID)
        var inputSize = inputSource.getInputDataSize(sourceChannelID)
        if (inputSize == nil) { return }
        
        //  Process each operator
        for networkOperator in networkOperators {
            inputs = networkOperator.feedForward(inputs, inputSize: inputSize!)
            inputSize = networkOperator.getResultSize()
        }
    }
    
    //  Function to clear weight-change accumulations for the start of a batch
    public func startBatch()
    {
        for networkOperator in networkOperators {
            networkOperator.startBatch()
        }
    }
    
    func backPropagate(_ gradientSource: DeepNetworkOutputDestination)
    {
        //  Get the gradients from the previous layer
        inputErrorGradient = gradientSource.getGradientForSource(idString)
        
        //  Process the gradient backwards through all the operators
        for operatorIndex in stride(from: (networkOperators.count - 1), through: 0, by: -1) {
            inputErrorGradient = networkOperators[operatorIndex].backPropogateGradient(inputErrorGradient)
        }
    }
    
    public func updateWeights(_ trainingRate : Float, weightDecay: Float)
    {
        for networkOperator in networkOperators {
            networkOperator.updateWeights(trainingRate, weightDecay: weightDecay)
        }
    }
    
    public func gradientCheck(ε: Float, Δ: Float, network: DeepNetwork) -> Bool
    {
        //  Have each operator check
        var result = true
        for operatorIndex in 0..<networkOperators.count {
            if (!networkOperators[operatorIndex].gradientCheck(ε: ε, Δ: Δ, network: network)) { result = false }
        }
        return result
    }

    func getGradient() -> [Float]
    {
        return inputErrorGradient
    }
    
    ///  Function to get the result of the last operation
    public func getFinalResult() -> [Float]
    {
        if let lastOperator = networkOperators.last {
            return lastOperator.getResults()
        }
        return []
    }
    
    public func getResultOfItem(_ operatorIndex: Int) ->(values : [Float], size: DeepChannelSize)?
    {
        if (operatorIndex >= 0 && operatorIndex < networkOperators.count) {
            let values = networkOperators[operatorIndex].getResults()
            let size = networkOperators[operatorIndex].getResultSize()
            return (values : values, size: size)
        }
        return nil
    }
    
    public func getPersistenceDictionary() -> [String: AnyObject]
    {
        var resultDictionary : [String: AnyObject] = [:]
        
        //  Set the id string type
        resultDictionary["idString"] = idString as AnyObject?
        
        //  Set the source ID
        resultDictionary["sourceChannelID"] = sourceChannelID as AnyObject?
        
        //  Set the array of network operators
        var operationsArray : [[String: AnyObject]] = []
        for networkOperator in networkOperators {
            operationsArray.append(networkOperator.getOperationPersistenceDictionary())
        }
        resultDictionary["networkOperators"] = operationsArray as AnyObject?
        
        return resultDictionary
    }

}
