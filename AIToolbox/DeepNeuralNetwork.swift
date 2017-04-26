//
//  DeepNeuralNetwork.swift
//  AIToolbox
//
//  Created by Kevin Coble on 7/1/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation


import Foundation
import Accelerate


final public class DeepNeuralNetwork : DeepNetworkOperator
{
    var activation : NeuralActivationFunction
    var numInputs = 0
    var numNodes : Int
    var resultSize : DeepChannelSize
    var weights : [Float] = []
    var lastNodeSums : [Float]
    var lastOutputs : [Float]
    fileprivate var inputsWithBias : [Float] = []
    var weightAccumulations : [Float] = []
    
    public init(activation : NeuralActivationFunction, size: DeepChannelSize)
    {
        self.activation = activation
        self.resultSize = size
        
        //  Get the number of nodes
        numNodes = resultSize.totalSize
        
        //  Allocate the arrays for results
        lastNodeSums = [Float](repeating: 0.0, count: numNodes)
        lastOutputs = [Float](repeating: 0.0, count: numNodes)
    }
    
    public init?(fromDictionary: [String: AnyObject])
    {
        //  Init for nil return (hopefully Swift 3 removes this need)
        resultSize = DeepChannelSize(dimensionCount: 0, dimensionValues: [])
        numNodes = 0
        weights = []
        lastNodeSums = []
        lastOutputs = []
        
        //  Get the activation type
        let activationTypeValue = fromDictionary["activation"] as? NSInteger
        if activationTypeValue == nil { return nil }
        let tempActivationType = NeuralActivationFunction(rawValue: activationTypeValue!)
        if (tempActivationType == nil) { return nil }
        activation = tempActivationType!
        
        //  Get the number of dimension
        let dimensionValue = fromDictionary["numDimension"] as? NSInteger
        if dimensionValue == nil { return nil }
        let numDimensions = dimensionValue!
        
        //  Get the dimensions levels
        let tempArray = getIntArray(fromDictionary, identifier: "dimensions")
        if (tempArray == nil) { return nil }
        let dimensions = tempArray!
        resultSize = DeepChannelSize(dimensionCount: numDimensions, dimensionValues: dimensions)
        
        //  Get the number of nodes
        numNodes = resultSize.totalSize
        
        //  Get the weights
        let tempWeights = getFloatArray(fromDictionary, identifier: "weights")
        if (tempWeights == nil) { return nil }
        weights = tempWeights!
        numInputs = (weights.count / numNodes) - 1
        
        //  Allocate the arrays for results
        lastNodeSums = [Float](repeating: 0.0, count: numNodes)
        lastOutputs = [Float](repeating: 0.0, count: numNodes)
    }
    
    public func getType() -> DeepNetworkOperatorType
    {
        return .feedForwardNetOperation
    }
    
    public func getDetails() -> String
    {
        var result = activation.getString() + " ["
        if (resultSize.numDimensions > 0) { result += "\(resultSize.dimensions[0])" }
        if (resultSize.numDimensions > 1) {
            for i in 1..<resultSize.numDimensions {
                result += ", \(resultSize.dimensions[i])"
            }
        }
        result += "]"
        return result
    }
    
    public func getResultingSize(_ inputSize: DeepChannelSize) -> DeepChannelSize
    {
        //  Input size does not affect output size.  However, it does change the weight sizing
        let newInputCount = inputSize.totalSize
        if (newInputCount != numInputs) {
            numInputs = newInputCount
            initializeParameters()
        }
        
        return resultSize
    }
    
    public func initializeParameters()
    {
        //  Allocate the weight array using 'Xavier' initialization
        let numWeights = (numInputs + 1) * numNodes   //  Add bias offset
        var weightDiviser: Float
        if (activation == .rectifiedLinear) {
            weightDiviser = 1 / sqrt(Float(numInputs) * 0.5)
        }
        else {
            weightDiviser = 1 / sqrt(Float(numInputs))
        }
        weights = []
        for _ in 0..<numWeights {
            weights.append(Gaussian.gaussianRandomFloat(0.0, standardDeviation : 1.0) * weightDiviser)
        }
    }
    
    public func feedForward(_ inputs: [Float], inputSize: DeepChannelSize) -> [Float]
    {
        //  Get inputs with a bias term
        inputsWithBias = inputs
        inputsWithBias.append(1.0)
        
        //  Multiply the weight matrix by the inputs to get the node sum values
        vDSP_mmul(weights, 1, inputsWithBias, 1, &lastNodeSums, 1, vDSP_Length(numNodes), 1, vDSP_Length(numInputs+1))
        
        //  Perform the non-linearity
        switch (activation) {
        case .none:
            lastOutputs = lastNodeSums
            break
        case .hyperbolicTangent:
            lastOutputs = lastNodeSums.map({ tanh($0) })
            break
        case .sigmoidWithCrossEntropy:
            fallthrough
        case .sigmoid:
            lastOutputs = lastNodeSums.map( { 1.0 / (1.0 + exp(-$0)) } )
            break
        case .rectifiedLinear:
            lastOutputs = lastNodeSums.map( { $0 < 0 ? 0.0 : $0 } )
            break
        case .softSign:
            lastOutputs = lastNodeSums.map( { $0 / (1.0 + exp($0)) } )
            break
        case .softMax:
            lastOutputs = lastNodeSums.map( { exp($0) } )
            break
        }
        
        return lastOutputs
    }
    
    public func getResults() -> [Float]
    {
        return lastOutputs
    }
    
    public func getResultSize() -> DeepChannelSize
    {
        return resultSize
    }
    
    public func getResultRange() ->(minimum: Float, maximum: Float)
    {
        if activation == .hyperbolicTangent {
            return (minimum: -1.0, maximum: 1.0)
        }
        return (minimum: 0.0, maximum: 1.0)
    }
    
    public func startBatch()
    {
        //  Clear the weight accumulations
        weightAccumulations = [Float](repeating: 0.0, count: weights.count)
    }
    
    //  𝟃E/𝟃h comes in, 𝟃E/𝟃x goes out
    public func backPropogateGradient(_ upStreamGradient: [Float]) -> [Float]
    {
        //  Forward equation is h = fn(Wx), where fn is the activation function
        //  The 𝟃E/𝟃h comes in, we need to calculate 𝟃E/𝟃W and 𝟃E/𝟃x
        //       𝟃E/𝟃W = 𝟃E/𝟃h ⋅ 𝟃h/𝟃z ⋅ 𝟃z/𝟃W
        //             = upStreamGradient ⋅ activation' ⋅ input
        
        //  Get 𝟃E/𝟃z
        var 𝟃E𝟃z : [Float]
        switch (activation) {
        case .none:
            𝟃E𝟃z = upStreamGradient
            break
        case .hyperbolicTangent:
            𝟃E𝟃z = upStreamGradient
            for index in 0..<lastOutputs.count {
                𝟃E𝟃z[index] *= (1 - lastOutputs[index] * lastOutputs[index])
            }
            break
        case .sigmoidWithCrossEntropy:
            fallthrough
        case .sigmoid:
            𝟃E𝟃z = upStreamGradient
            for index in 0..<lastOutputs.count {
                𝟃E𝟃z[index] *= (lastOutputs[index] - (lastOutputs[index] * lastOutputs[index]))
            }
            break
        case .rectifiedLinear:
            𝟃E𝟃z = upStreamGradient
            for index in 0..<lastOutputs.count {
                if (lastOutputs[index] < 0.0) { 𝟃E𝟃z[index] = 0.0 }
            }
            break
        case .softSign:
            𝟃E𝟃z = upStreamGradient
            var z : Float
            //  Reconstitute z from h
            for index in 0..<lastOutputs.count {
                if (lastOutputs[index] < 0) {        //  Negative z
                    z = lastOutputs[index] / (1.0 + lastOutputs[index])
                    𝟃E𝟃z[index] /= -((1.0 + z) * (1.0 + z))
                }
                else {              //  Positive z
                    z = lastOutputs[index] / (1.0 - lastOutputs[index])
                    𝟃E𝟃z[index] /= ((1.0 + z) * (1.0 + z))
                }
            }
            break
        case .softMax:
            //  Should not get here - softmax is not allowed except on final layer
            𝟃E𝟃z = upStreamGradient
            break
        }
        
        //  Get 𝟃E/𝟃W.  𝟃E/𝟃W = 𝟃E/𝟃z ⋅ 𝟃z/𝟃W = 𝟃E𝟃z ⋅ inputsWithBias
        var weightChange = [Float](repeating: 0.0, count: weights.count)
        vDSP_mmul(𝟃E𝟃z, 1, inputsWithBias, 1, &weightChange, 1, vDSP_Length(numNodes), vDSP_Length(numInputs+1), 1)
        vDSP_vadd(weightChange, 1, weightAccumulations, 1, &weightAccumulations, 1, vDSP_Length(weightChange.count))

        
        //  Get 𝟃E/𝟃x.  𝟃E/𝟃x = 𝟃E/𝟃z ⋅ 𝟃z/𝟃x = 𝟃E𝟃z ⋅ weights
//        var downStreamGradient = [Float](repeating: 0.0, count: numInputs)
//        for index in 0..<numInputs {
//            vDSP_dotpr(&weights[index], vDSP_Stride(numInputs+1), 𝟃E𝟃z, 1, &downStreamGradient[index], vDSP_Length(numNodes))
//        }
        
        var downStreamGradient = [Float](repeating: 0.0, count: numInputs+1)        //  an extra for the bias term in the weights
        vDSP_mmul(𝟃E𝟃z, 1, weights, 1, &downStreamGradient, 1, 1, vDSP_Length(numInputs+1), vDSP_Length(numNodes))
        downStreamGradient = Array(downStreamGradient[0..<numNodes])        //  Size for the inputs without the bias
        
        return downStreamGradient
    }
    
    public func updateWeights(_ trainingRate : Float, weightDecay: Float)
    {
        //  If there is a decay factor, use it
        if (weightDecay != 1.0) {
            var λ = weightDecay     //  Needed for unsafe pointer conversion
            vDSP_vsmul(weights, 1, &λ, &weights, 1, vDSP_Length(weights.count))
        }
        
        //  Subtract the weight changes from the weight matrix (W = W - η∇)
        var η = -trainingRate     //  Needed for unsafe pointer conversion
        vDSP_vsma(weightAccumulations, 1, &η, weights, 1, &weights, 1, vDSP_Length(weights.count))
    }
    
    public func gradientCheck(ε: Float, Δ: Float, network: DeepNetwork) -> Bool
    {
        var result = true
        
        //  Iterate through each parameter
        for index in 0..<weights.count {
            let oldValue = weights[index]
            
            //  Get the network loss with a small addition to the parameter
            weights[index] += ε
            network.feedForward()
            let plusLoss = network.getResultLoss()
            
            //  Get the network loss with a small subtraction from the parameter
            weights[index] = oldValue - ε
            network.feedForward()
            let minusLoss = network.getResultLoss()
            weights[index] = oldValue
            
            //  Iterate over the results
            for resultIndex in 0..<plusLoss.count {
                //  Get the numerical gradient estimate  𝟃E/𝟃W
                let gradient = (plusLoss[resultIndex] - minusLoss[resultIndex]) / (2.0 * ε)
                
                //  Compare with the analytical gradient
                let difference = abs(gradient - weightAccumulations[index])
                if (difference > Δ) { result = false }
            }
        }
        
        return result
    }

    
    public func getPersistenceDictionary() -> [String: AnyObject]
    {
        var resultDictionary : [String: AnyObject] = [:]
        
        //  Set the activation type
        resultDictionary["activation"] = activation.rawValue as AnyObject?
        
        //  Set the number of dimension
        resultDictionary["numDimension"] = resultSize.numDimensions as AnyObject?
        
        //  Set the dimensions levels
        resultDictionary["dimensions"] = resultSize.dimensions as AnyObject?
        
        //  Set the weights
        resultDictionary["weights"] = weights as AnyObject?
        
        return resultDictionary
    }
}
