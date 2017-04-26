//
//  DeepNonLinearity.swift
//  AIToolbox
//
//  Created by Kevin Coble on 4/13/17.
//  Copyright © 2017 Kevin Coble. All rights reserved.
//

import Foundation
import Accelerate


final public class DeepNonLinearity : DeepNetworkOperator
{
    public var activation : NeuralActivationFunction
    var lastOutputs : [Float]
    var resultSize : DeepChannelSize
    
    public init(activation : NeuralActivationFunction)
    {
        self.activation = activation
        self.resultSize = DeepChannelSize(dimensionCount: 0, dimensionValues: [])
        lastOutputs = []
    }
    
    public init?(fromDictionary: [String: AnyObject])
    {
        lastOutputs = []
        resultSize = DeepChannelSize(dimensionCount: 0, dimensionValues: [])
        
        //  Get the activation type
        let activationTypeValue = fromDictionary["activation"] as? NSInteger
        if activationTypeValue == nil { return nil }
        let tempActivationType = NeuralActivationFunction(rawValue: activationTypeValue!)
        if (tempActivationType == nil) { return nil }
        activation = tempActivationType!
    }
    
    public func getType() -> DeepNetworkOperatorType
    {
        return .nonLinearityOperation
    }
    
    public func getDetails() -> String
    {
        return activation.getString()
    }
    
    public func getResultingSize(_ inputSize: DeepChannelSize) -> DeepChannelSize
    {
        return inputSize
    }
    
    public func initializeParameters()
    {
        //  No parameters
    }
    
    public func feedForward(_ inputs: [Float], inputSize: DeepChannelSize) -> [Float]
    {
        //  Allocate an output array
        resultSize = inputSize
        lastOutputs = [Float](repeating: 0.0, count: inputSize.totalSize)
        
        //  Perform the non-linearity
        switch (activation) {
        case .none:
            lastOutputs = inputs
            break
        case .hyperbolicTangent:
            lastOutputs = inputs.map({ tanh($0) })
            break
        case .sigmoidWithCrossEntropy:
            fallthrough
        case .sigmoid:
            lastOutputs = inputs.map( { 1.0 / (1.0 + exp(-$0)) } )
            break
        case .rectifiedLinear:
            lastOutputs = inputs.map( { $0 < 0 ? 0.0 : $0 } )
            break
        case .softSign:
            lastOutputs = inputs.map( { $0 / (1.0 + exp($0)) } )
            break
        case .softMax:
            lastOutputs = inputs.map( { exp($0) } )
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
        //  No parameters
    }
    
    
    //  𝟃E/𝟃o comes in, 𝟃E/𝟃i goes out
    public func backPropogateGradient(_ upStreamGradient: [Float]) -> [Float]
    {
        //  Forward equation is o = fn(i) for each element of the input, where fn is the activation function
        //  The 𝟃E/𝟃o comes in, we need to calculate 𝟃E/𝟃i
        //       𝟃E/𝟃i = 𝟃E/𝟃o ⋅ 𝟃o/𝟃i
        //             = upStreamGradient ⋅ activation'
        
        //  Allocate the downstream gradient array
        var downStreamGradient = upStreamGradient
        
        //  Get 𝟃E/𝟃i for each element
        switch (activation) {
        case .none:
            break
        case .hyperbolicTangent:
            for index in 0..<lastOutputs.count {
                downStreamGradient[index] *= (1 - lastOutputs[index] * lastOutputs[index])
            }
            break
        case .sigmoidWithCrossEntropy:
            fallthrough
        case .sigmoid:
            for index in 0..<lastOutputs.count {
                downStreamGradient[index] *= (lastOutputs[index] - (lastOutputs[index] * lastOutputs[index]))
            }
            break
        case .rectifiedLinear:
            for index in 0..<lastOutputs.count {
                if (lastOutputs[index] < 0.0) { downStreamGradient[index] = 0.0 }
            }
            break
        case .softSign:
            var i : Float
            //  Reconstitute i from o
            for index in 0..<lastOutputs.count {
                if (lastOutputs[index] < 0) {        //  Negative i
                    i = lastOutputs[index] / (1.0 + lastOutputs[index])
                    downStreamGradient[index] /= -((1.0 + i) * (1.0 + i))
                }
                else {              //  Positive i
                    i = lastOutputs[index] / (1.0 - lastOutputs[index])
                    downStreamGradient[index] /= ((1.0 + i) * (1.0 + i))
                }
            }
            break
        case .softMax:
            //  Should not get here - softmax is not allowed except on final layer
            break
        }
        
        return downStreamGradient
    }
    
    public func updateWeights(_ trainingRate : Float, weightDecay: Float)
    {
        //  No parameters
    }
    
    public func gradientCheck(ε: Float, Δ: Float, network: DeepNetwork) -> Bool
    {
        //  No (stored) gradients in a nonlinearity operation
        return true
    }
    
    public func getPersistenceDictionary() -> [String: AnyObject]
    {
        var resultDictionary : [String: AnyObject] = [:]
        
        //  Set the activation type
        resultDictionary["activation"] = activation.rawValue as AnyObject?
        
        return resultDictionary
    }
}
 
