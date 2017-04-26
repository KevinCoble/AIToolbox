//
//  DeepNetworkOperator.swift
//  AIToolbox
//
//  Created by Kevin Coble on 6/26/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation


public enum DeepNetworkOperatorType : Int
{
    case convolution2DOperation = 0
    case poolingOperation
    case feedForwardNetOperation
    case nonLinearityOperation
    
    public func getString() ->String
    {
        switch self {
        case .convolution2DOperation:
            return "2D Convolution"
        case .poolingOperation:
            return "Pooling"
        case .feedForwardNetOperation:
            return "FeedForward NN"
        case .nonLinearityOperation:
            return "NonLinearity"
        }
    }
    
    public static func getAllTypes() -> [(name: String, type: DeepNetworkOperatorType)]
    {
        var raw = 0
        var results : [(name: String, type: DeepNetworkOperatorType)] = []
        while let type = DeepNetworkOperatorType(rawValue: raw) {
            results.append((name: type.getString(), type: type))
            raw += 1
        }
        return results
    }
    
    public static func getDeepNetworkOperatorFromDict(_ sourceDictionary: [String: AnyObject]) -> DeepNetworkOperator?
    {
        if let type = sourceDictionary["operatorType"] as? NSInteger {
            if let opType = DeepNetworkOperatorType(rawValue: type) {
                if let opDefinition = sourceDictionary["operatorDefinition"] as? [String: AnyObject] {
                    switch opType {
                    case .convolution2DOperation:
                        return Convolution2D(fromDictionary: opDefinition)
                    case .poolingOperation:
                        return Pooling(fromDictionary: opDefinition)
                    case .feedForwardNetOperation:
                        return DeepNeuralNetwork(fromDictionary: opDefinition)
                    case .nonLinearityOperation:
                        return DeepNonLinearity(fromDictionary: opDefinition)
                    }
                }
            }
        }
        return nil
    }
}


public protocol DeepNetworkOperator : MLPersistence {
    func getType() -> DeepNetworkOperatorType
    func getDetails() -> String
    func getResultingSize(_ inputSize: DeepChannelSize) -> DeepChannelSize
    func initializeParameters()
    func feedForward(_ inputs: [Float], inputSize: DeepChannelSize) -> [Float]
    func getResults() -> [Float]
    func getResultSize() -> DeepChannelSize
    func getResultRange() ->(minimum: Float, maximum: Float)
    func startBatch()
    func backPropogateGradient(_ upStreamGradient: [Float]) -> [Float]
    func updateWeights(_ trainingRate : Float, weightDecay: Float)
    func gradientCheck(ε: Float, Δ: Float, network: DeepNetwork) -> Bool
}


extension DeepNetworkOperator {
    
    public func getOperationPersistenceDictionary() -> [String : AnyObject] {
        var resultDictionary : [String: AnyObject] = [:]
        
        //  Set the operator type
        resultDictionary["operatorType"] = getType().rawValue as AnyObject?
        
        //  Set the definition
        resultDictionary["operatorDefinition"] = getPersistenceDictionary() as AnyObject?
        
        return resultDictionary
    }
}
