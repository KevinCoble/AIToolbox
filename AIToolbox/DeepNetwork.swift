//
//  DeepNetwork.swift
//  AIToolbox
//
//  Created by Kevin Coble on 6/25/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation

protocol DeepNetworkInputSource {
    func getInputDataSize(_ inputIDs : [String]) -> DeepChannelSize?
    func getValuesForIDs(_ inputIDs : [String]) -> [Float]
    func getAllValues() -> [Float]
}

protocol DeepNetworkOutputDestination {
    func getGradientForSource(_ sourceID : String) -> [Float]
}


public struct DeepNetworkInput : MLPersistence {
    public let inputID : String
    public private(set) var size : DeepChannelSize
    var values : [Float]
    
    public init(inputID: String, size: DeepChannelSize, values: [Float])
    {
        self.inputID = inputID
        self.size = size
        self.values = values
    }
    
    //  Persistence assumes values are set later, and only ID and size need to be saved
    public init?(fromDictionary: [String: AnyObject])
    {
        //  Init for nil return (hopefully Swift 3 removes this need)
        size = DeepChannelSize(dimensionCount: 0, dimensionValues: [])
        values = []
        
        //  Get the id string
        let id = fromDictionary["inputID"] as? NSString
        if id == nil { return nil }
        inputID = id! as String
        
        //  Get the number of dimension
        let dimensionValue = fromDictionary["numDimension"] as? NSInteger
        if dimensionValue == nil { return nil }
        let numDimensions = dimensionValue!
        
        //  Get the dimensions values
        let tempArray = getIntArray(fromDictionary, identifier: "dimensions")
        if (tempArray == nil) { return nil }
        let dimensions = tempArray!
        size = DeepChannelSize(dimensionCount: numDimensions, dimensionValues: dimensions)
    }
    
    public func getPersistenceDictionary() -> [String: AnyObject]
    {
        var resultDictionary : [String: AnyObject] = [:]
        
        //  Set the identifier
        resultDictionary["inputID"] = inputID as AnyObject?
        
        //  Set the number of dimension
        resultDictionary["numDimension"] = size.numDimensions as AnyObject?
        
        //  Set the dimensions levels
        resultDictionary["dimensions"] = size.dimensions as AnyObject?
        
        return resultDictionary
    }
}

///  Top-level class for a deep neural network definition
final public class DeepNetwork : DeepNetworkInputSource, DeepNetworkOutputDestination, MLPersistence
{
    var inputs : [DeepNetworkInput] = []
    
    var layers : [DeepLayer] = []
    
    var validated = false
    
    fileprivate var finalResultMin : Float = 0.0
    fileprivate var finalResultMax : Float = 1.0
    
    fileprivate var finalResults : [Float] = []
    fileprivate var errorVector : [Float] = []
    fileprivate var expectedClass : Int = 0
    
    public init() {
    }
    
    public init?(fromDictionary: [String: AnyObject])
    {
        //  Get the array of inputs
        let inputArray = fromDictionary["inputs"] as? NSArray
        if (inputArray == nil)  { return nil }
        for item in inputArray! {
            let element = item as? [String: AnyObject]
            if (element == nil)  { return nil }
            let input = DeepNetworkInput(fromDictionary: element!)
            if (input == nil)  { return nil }
            inputs.append(input!)
        }
        
        //  Get the array of layers
        let layerArray = fromDictionary["layers"] as? NSArray
        if (layerArray == nil)  { return nil }
        for item in layerArray! {
            let element = item as? [String: AnyObject]
            if (element == nil)  { return nil }
            let layer = DeepLayer(fromDictionary: element!)
            if (layer == nil)  { return nil }
            layers.append(layer!)
        }
    }
    
    ///  Function to get the number of defined inputs
    public var numInputs: Int {
        get { return inputs.count }
    }
    
    ///  Function to get the input for the specified index
    public func getInput(atIndex: Int) -> DeepNetworkInput
    {
        return inputs[atIndex]
    }
    
    ///  Function to add an input to the network
    public func addInput(_ newInput: DeepNetworkInput)
    {
        inputs.append(newInput)
    }
    
    ///  Function to remove an input from the network
    public func removeInput(_ inputIndex: Int)
    {
        if (inputIndex >= 0 && inputIndex < inputs.count) {
            inputs.remove(at: inputIndex)
        }
        validated = false
    }
    
    //  Function to get the index from an input ID
    public func getInputIndex(_ idString: String) -> Int?
    {
        for index in 0..<inputs.count {
            if (inputs[index].inputID == idString) { return index }
        }
        return nil
    }
    
    ///  Function to set the values for an input
    public func setInputValues(_ forInput: String, values : [Float])
    {
        //  Get the input to process
        if let index = getInputIndex(forInput) {
            inputs[index].values = values
        }
    }
    
    ///  Function to get the size of an input set
    public func getInputDataSize(_ inputIDs : [String]) -> DeepChannelSize?
    {
        var currentSize : DeepChannelSize?
        
        //  Get the size of each input
        for sourceID in inputIDs {
            if let inputIndex = getInputIndex(sourceID) {
                if (currentSize == nil) {
                    //  First input - start with this size
                    currentSize = inputs[inputIndex].size
                }
                else {
                    //  Not first input, add sizes together
                    let extendedCurrentSize = currentSize!.dimensions + [1, 1, 1]
                    let extendedInputSize = inputs[inputIndex].size.dimensions + [1, 1, 1]
                    for index in 0..<4 {
                        if (index <= currentSize!.numDimensions && index <= inputs[index].size.numDimensions) {
                            if (extendedCurrentSize[index] != extendedInputSize[index]) { return nil }
                            continue
                        }
                        if (index == currentSize!.numDimensions && index == inputs[inputIndex].size.numDimensions) {
                            var newDimensions = currentSize!.dimensions
                            newDimensions.append(extendedCurrentSize[index] + extendedInputSize[index])
                            currentSize = DeepChannelSize(dimensionCount : index + 1, dimensionValues : newDimensions)
                            break
                        }
                        else if (index < currentSize!.numDimensions) {
                            var newDimensions = currentSize!.dimensions
                            newDimensions[index] = (extendedCurrentSize[index] + extendedInputSize[index])
                            currentSize = DeepChannelSize(dimensionCount : index + 1, dimensionValues : newDimensions)
                            break
                        }
                        else {
                            var newDimensions = inputs[inputIndex].size.dimensions
                            newDimensions[index] = (extendedCurrentSize[index] + extendedInputSize[index])
                            currentSize = DeepChannelSize(dimensionCount : index + 1, dimensionValues : newDimensions)
                            break
                        }
                    }
                }
            }
            else {
                //  The input source wasn't found
                return nil
            }
        }
        
        return currentSize
    }

    
    ///  Function to get the number of defined layers
    public var numLayers: Int {
        get { return layers.count }
    }
    
    ///  Function to get the layer for the specified index
    public func getLayer(atIndex: Int) -> DeepLayer
    {
        return layers[atIndex]
    }
    
    ///  Function to add a layer to the network
    public func addLayer(_ newLayer: DeepLayer)
    {
        layers.append(newLayer)
        validated = false
    }
    
    ///  Function to remove a layer from the network
    public func removeLayer(_ layerIndex: Int)
    {
        if (layerIndex >= 0 && layerIndex < layers.count) {
            layers.remove(at: layerIndex)
            validated = false
        }
    }
    
    ///  Function to add a channel to the network
    public func addChannel(_ toLayer: Int, newChannel: DeepChannel)
    {
        if (toLayer >= 0 && toLayer < layers.count) {
            layers[toLayer].addChannel(newChannel)
            validated = false
        }
    }
    
    ///  Functions to remove a channel from the network
    public func removeChannel(_ layer: Int, channelIndex: Int)
    {
        if (layer >= 0 && layer < layers.count) {
            layers[layer].removeChannel(channelIndex)
            validated = false
        }
    }
    public func removeChannel(_ layer: Int, channelID: String)
    {
        if let index = layers[layer].getChannelIndex(channelID) {
            removeChannel(layer, channelIndex: index)
        }
    }
    
    ///  Functions to add a network operator to the network
    public func addNetworkOperator(_ toLayer: Int, channelIndex: Int, newOperator: DeepNetworkOperator)
    {
        if (toLayer >= 0 && toLayer < layers.count) {
            layers[toLayer].addNetworkOperator(channelIndex, newOperator: newOperator)
            validated = false
        }
    }
    public func addNetworkOperator(_ toLayer: Int, channelID: String, newOperator: DeepNetworkOperator)
    {
        if (toLayer >= 0 && toLayer < layers.count) {
            if let index = layers[toLayer].getChannelIndex(channelID) {
                layers[toLayer].addNetworkOperator(index, newOperator: newOperator)
                validated = false
            }
        }
    }
    
    ///  Function to get the network operator at the specified index
    public func getNetworkOperator(_ toLayer: Int, channelIndex: Int, operatorIndex: Int) ->DeepNetworkOperator?
    {
        if (toLayer >= 0 && toLayer < layers.count) {
            return layers[toLayer].getNetworkOperator(channelIndex, operatorIndex: operatorIndex)
        }
        
        return nil
    }
    public func getNetworkOperator(_ toLayer: Int, channelID: String, operatorIndex: Int) ->DeepNetworkOperator?
    {
        if (toLayer >= 0 && toLayer < layers.count) {
            if let index = layers[toLayer].getChannelIndex(channelID) {
                return layers[toLayer].getNetworkOperator(index, operatorIndex: operatorIndex)
            }
        }
        
        return nil
    }
    
    ///  Functions to replace a network operator at the specified index
    public func replaceNetworkOperator(_ toLayer: Int, channelIndex: Int, operatorIndex: Int, newOperator: DeepNetworkOperator)
    {
        if (toLayer >= 0 && toLayer < layers.count) {
            layers[toLayer].replaceNetworkOperator(channelIndex, operatorIndex: operatorIndex, newOperator: newOperator)
        }
    }
    public func replaceNetworkOperator(_ toLayer: Int, channelID: String, operatorIndex: Int, newOperator: DeepNetworkOperator)
    {
        if (toLayer >= 0 && toLayer < layers.count) {
            if let index = layers[toLayer].getChannelIndex(channelID) {
                layers[toLayer].replaceNetworkOperator(index, operatorIndex: operatorIndex, newOperator: newOperator)
            }
        }
    }
    
    ///  Functions to remove a network operator from the network
    public func removeNetworkOperator(_ layer: Int, channelIndex: Int, operatorIndex: Int)
    {
        if (layer >= 0 && layer < layers.count) {
            layers[layer].removeNetworkOperator(channelIndex, operatorIndex: operatorIndex)
            validated = false
        }
    }
    public func removeNetworkOperator(_ layer: Int, channelID: String, operatorIndex: Int)
    {
        if let index = layers[layer].getChannelIndex(channelID) {
            layers[layer].removeNetworkOperator(index, operatorIndex: operatorIndex)
            validated = false
        }
    }
    
    ///  Function to validate a DeepNetwork
    ///  This method checks that the inputs to each layer are available, returning an array of strings describing any errors
    ///  The resulting size of each channel is updated as well
    public func validateNetwork() -> [String]
    {
        var errorStrings: [String] = []
        
        var prevLayer : DeepNetworkInputSource = self
        
        for layer in 0..<layers.count {
            errorStrings += layers[layer].validateAgainstPreviousLayer(prevLayer, layerIndex: layer)
            prevLayer = layers[layer]
        }
        
        validated = (errorStrings.count == 0)
        
        finalResultMin = 0.0
        finalResultMax = 1.0
        if let lastLayer = layers.last {
            let range = lastLayer.getResultRange()
            finalResultMin = range.minimum
            finalResultMax = range.maximum
        }
        
        return errorStrings
    }
    
    
    ///  Property to get if the network is validated
    public var isValidated: Bool {
        get { return validated }
    }

    
    ///  Function to set all learnable parameters to random initialization again
    public func initializeParameters()
    {
        //  Have each layer initialize
        for layer in 0..<layers.count {
            layers[layer].initializeParameters()
        }
    }
    
    ///  Function to run the network forward.  Assumes inputs have been set
    ///  Returns the values from the last layer
    @discardableResult
    public func feedForward() -> [Float]
    {
        //  Make sure we are validated
        if (!validated) {
            _ = validateNetwork()
            if !validated { return [] }
        }
        
        var prevLayer : DeepNetworkInputSource = self
        
        for layer in 0..<layers.count {
            layers[layer].feedForward(prevLayer)
            prevLayer = layers[layer]
        }
        
        //  Keep track of the number of outputs from the final layer, so we can generate an error term later
        finalResults = prevLayer.getAllValues()
        
        return finalResults
    }
    
    public func getResultClass() -> Int
    {
        if finalResults.count == 1 {
            if finalResults[0] > ((finalResultMax + finalResultMin) * 0.5) { return 1 }
            return 0
        }
        else {
            var bestClass = 0
            var highestResult = -Float.infinity
            for classIndex in 0..<finalResults.count {
                if (finalResults[classIndex] > highestResult) {
                    highestResult = finalResults[classIndex]
                    bestClass = classIndex
                }
            }
            return bestClass
        }
    }
    
    ///  Function to clear weight-change accumulations for the start of a batch
    public func startBatch()
    {
        for layer in layers {
            layer.startBatch()
        }
    }
    
    ///  Function to run the network backward using an expected output array, propagating the error term.  Assumes inputs have been set
    public func backPropagate(_ expectedResults: [Float])
    {
        if (layers.count < 1) { return }  //  Can't backpropagate through non-existant layers
        
        //  Get an error vector to pass to the final layer.  This is the gradient if using least-squares error
        getErrorVector(expectedOutput: expectedResults)
        
        //  Propagate to the last layer
        layers.last!.backPropagate(self)
        
        //  Propogate to all previous layers
        for layerIndex in stride(from: (layers.count - 2), through: 0, by: -1) {
            layers[layerIndex].backPropagate(layers[layerIndex+1])
        }
    }
    
    ///  Function to run the network backward using an expected class output, propagating the error term.  Assumes inputs have been set
    public func backPropagate(_ expectedResultClass: Int)
    {
        //  Remember the expected result class
        expectedClass = expectedResultClass
        
        if (layers.count < 1) { return }  //  Can't backpropagate through non-existant layers
            
        //  Get an error vector to pass to the final layer.  This is the gradient if using least-squares error
        getErrorVector()
        
        //  Propagate to the last layer
        layers.last!.backPropagate(self)
        
        //  Propogate to all previous layers
        for layerIndex in stride(from: (layers.count - 2), through: 0, by: -1) {
            layers[layerIndex].backPropagate(layers[layerIndex+1])
        }
    }
    
    //  Get the error vector - 𝟃E/𝟃o:  derivitive of error with respect to output - from expected class
    func getErrorVector()
    {
        //  Get an error vector to pass to the final layer.  This is the gradient if using least-squares error
        //  E = 0.5(output - expected)²  -->  𝟃E/𝟃o = output - expected
        errorVector = [Float](repeating: finalResultMin, count: finalResults.count)
        if (finalResults.count == 1) {
            if (expectedClass == 1) {
                errorVector[0] = finalResults[0] - finalResultMax
            }
            else {
                errorVector[0] = finalResults[0] - finalResultMin
            }
        }
        else {
            for index in 0..<finalResults.count {
                if (index == expectedClass) {
                    errorVector[index] =  finalResults[index] - finalResultMax
                }
                else {
                    errorVector[index] = finalResults[index] - finalResultMin
                }
            }
        }
    }
    
    //  Get the error vector - 𝟃E/𝟃o:  derivitive of error with respect to output - from expected output array
    func getErrorVector(expectedOutput: [Float])
    {
        //  Get an error vector to pass to the final layer.  This is the gradient if using least-squares error
        //  E = 0.5(output - expected)²  -->  𝟃E/𝟃o = output - expected
        errorVector = [Float](repeating: finalResultMin, count: finalResults.count)
        for index in 0..<finalResults.count {
            errorVector[index] = finalResults[index] - expectedOutput[index]
        }
    }
    
    ///  Function to get the expected network output for a given class
    public func getExpectedOutput(_ forClass: Int) ->[Float]
    {
        var expectedOutputVector = [Float](repeating: finalResultMin, count: finalResults.count)
        if (finalResults.count == 1) {
            if (forClass == 1) { expectedOutputVector[0] = finalResultMax }
        }
        else {
            if (forClass >= 0 && forClass < finalResults.count) {
                expectedOutputVector[forClass] = finalResultMax
            }
        }
        
        return expectedOutputVector
    }
    
    ///  Function to get the total error with respect to an expected output class
    public func getTotalError(_ expectedClass: Int) ->Float
    {
        var result : Float = 0.0
        
        let expectedOutput = getExpectedOutput(expectedClass)
        for index in 0..<expectedOutput.count {
            result += abs(expectedOutput[index] - finalResults[index])
        }
        
        return result
    }
    
    ///  Function to get the total error with respect to an expected output vector
    public func getTotalError(_ expectedOutput: [Float]) ->Float
    {
        var result : Float = 0.0
        
        for index in 0..<expectedOutput.count {
            result += abs(expectedOutput[index] - finalResults[index])
        }
        
        return result
    }
    
    public func updateWeights(_ trainingRate : Float, weightDecay: Float)
    {
        for layer in layers {
            layer.updateWeights(trainingRate, weightDecay: weightDecay)
        }
    
    }
    
    ///  Function to perform a gradient check on the network
    ///  Assumes forward pass and back-propogation have been performed already
    public func gradientCheck(ε: Float, Δ: Float) -> Bool
    {
        //  Have each layer check their gradients
        var result = true
        for layer in 0..<layers.count {
            if (!layers[layer].gradientCheck(ε: ε, Δ: Δ, network: self)) { result = false }
        }
        return result
    }
    
    public func getResultLoss() -> [Float]
    {
        let resultCount = finalResults.count
        var loss = [Float](repeating: 0.0, count: resultCount)
        
        //  Get the error vector
        getErrorVector()
        
        //  Assume squared error loss
        for i in 0..<resultCount {
            loss[i] = errorVector[i] * errorVector[i] * 0.5
        }
        
        return loss
    }
    
    public func getValuesForIDs(_ inputIDs : [String]) -> [Float]
    {
        var combinedInputs : [Float] = []
        
        //  Get the index
        for sourceID in inputIDs {
            if let index = getInputIndex(sourceID) {
                combinedInputs += inputs[index].values
            }
            else {
                return []
            }
        }
        
        return combinedInputs
    }
    
    public func getAllValues() -> [Float]
    {
        var result : [Float] = []
        for input in inputs {
            result += input.values
        }
        return result
    }
    
    func getGradientForSource(_ sourceID : String) -> [Float]
    {
        //  We only have one 'channel', so just return the error
        return errorVector
    }

    public func getResultOfItem(_ layer: Int, channelIndex: Int, operatorIndex: Int) ->(values : [Float], size: DeepChannelSize)?
    {
        if (layer >= 0 && layer < layers.count) {
            return layers[layer].getResultOfItem(channelIndex, operatorIndex: operatorIndex)
        }
        return nil
    }
    
    public func getPersistenceDictionary() -> [String: AnyObject]
    {
        var resultDictionary : [String: AnyObject] = [:]
        
        //  Set the array of inputs
        var inputArray : [[String: AnyObject]] = []
        for input in inputs {
            inputArray.append(input.getPersistenceDictionary())
        }
        resultDictionary["inputs"] = inputArray as AnyObject?
        
        //  Set the array of layers
        var layerArray : [[String: AnyObject]] = []
        for layer in layers {
            layerArray.append(layer.getPersistenceDictionary())
        }
        resultDictionary["layers"] = layerArray as AnyObject?
        
        return resultDictionary
    }
}
