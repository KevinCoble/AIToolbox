//
//  Pooling.swift
//  Convolution
//
//  Created by Kevin Coble on 2/20/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation
import Accelerate

public enum PoolingType : Int {
    case average = 0
    case minimum
    case maximum
    
    public func getString() ->String
    {
        switch self {
        case .average:
            return "Average"
        case .minimum:
            return "Minimum"
        case .maximum:
            return "Maximum"
        }
    }
}

final public class Pooling : DeepNetworkOperator
{
    public private(set) var poolType : PoolingType
    public private(set) var dimension: Int
    public private(set) var reductionLevels: [Int]
    var pool : [Float] = []
    var resultSize : DeepChannelSize
    fileprivate var inputSize = DeepChannelSize(dimensionCount: 0, dimensionValues: [])
    var inputUsed: [Int]?
    
    public init(type : PoolingType, dimension: Int)
    {
        poolType = type
        self.dimension = dimension      //  Max of 4 at this time - we will add error handling later
        reductionLevels = [Int](repeating: 1, count: dimension)
        resultSize = DeepChannelSize(dimensionCount: 0, dimensionValues: [])
    }
    
    public init(type : PoolingType, reduction: [Int])
    {
        poolType = type
        self.dimension = reduction.count      //  Max of 4 at this time - we will add error handling later
        reductionLevels = reduction
        resultSize = DeepChannelSize(dimensionCount: 0, dimensionValues: [])
    }
    
    public init?(fromDictionary: [String: AnyObject])
    {
        //  Init for nil return (hopefully Swift 3 removes this need)
        reductionLevels = []
        resultSize = DeepChannelSize(dimensionCount: 0, dimensionValues: [])
        
        //  Get the pooling type
        let poolTypeValue = fromDictionary["poolingType"] as? NSInteger
        if poolTypeValue == nil { return nil }
        let tempPoolType = PoolingType(rawValue: poolTypeValue!)
        if (tempPoolType == nil) { return nil }
        poolType = tempPoolType!
        
        //  Get the dimension
        let dimensionValue = fromDictionary["dimension"] as? NSInteger
        if dimensionValue == nil { return nil }
        dimension = dimensionValue!
        
        //  Get the reduction levels
        let tempArray = getIntArray(fromDictionary, identifier: "reductionLevels")
        if (tempArray == nil) { return nil }
        reductionLevels = tempArray!
        
        resultSize = DeepChannelSize(dimensionCount: dimension, dimensionValues: reductionLevels)
    }
    
    public func setReductionLevel(_ forDimension: Int, newLevel: Int)
    {
        if (forDimension >= 0 && forDimension < dimension) {
            reductionLevels[forDimension] = newLevel
        }
    }
    
    public func getType() -> DeepNetworkOperatorType
    {
        return .poolingOperation
    }
    
    public func getDetails() -> String
    {
        var result : String
        switch poolType {
        case .average:
            result = "Avg ["
        case .minimum:
            result = "Min ["
        case .maximum:
            result = "Max ["
        }
        if (dimension > 0) { result += "\(reductionLevels[0])" }
        if (dimension > 1) {
            for i in 1..<dimension {
                result += ", \(reductionLevels[i])"
            }
        }
        result += "]"
        return result
    }
    
    public func getResultingSize(_ inputSize: DeepChannelSize) -> DeepChannelSize
    {
        //  Reduce each of the dimensions by the specified reduction levels
        resultSize = inputSize
        for i in 0..<dimension {
            resultSize.dimensions[i] /= reductionLevels[i]
        }
        
        return resultSize
    }
    
    public func initializeParameters()
    {
        //  No parameters to initialize in a pooling layer
    }
    
    public func feedForward(_ inputs: [Float], inputSize: DeepChannelSize) -> [Float]
    {
        self.inputSize = inputSize
        
        //  Limit reduction to a 1 pixel value in each dimension
        var sourceSize = inputSize.dimensions
        sourceSize += [1, 1, 1]     //  Add size for missing dimensions
        var resultSize = inputSize.dimensions
        resultSize += [1, 1, 1]     //  Add size for missing dimensions
        var reduction = [Int](repeating: 1, count: 4)
        var sourceStride = [Int](repeating: 1, count: 4)
        var resultStride = [Int](repeating: 1, count: 4)
        var totalSize = 1
        for index in 0..<dimension {
            reduction[index] = reductionLevels[index]
            if (inputSize.dimensions[index]  < reduction[index]) { reduction[index] = inputSize.dimensions[index] }
            resultSize[index] = inputSize.dimensions[index] / reductionLevels[index]
            totalSize *= resultSize[index]
        }
        
        //  Determine the stride for each dimension
        for index in 0..<4 {
            if (index > 0) {
                for i in 0..<index { sourceStride[index] *= sourceSize[i] }
                for i in 0..<index { resultStride[index] *= resultSize[i] }
            }
        }
        
        //  Allocate the result array
        switch poolType {
        case .minimum:
            pool = [Float](repeating: Float.infinity, count: totalSize)
            inputUsed = [Int](repeating: 0, count: totalSize)
        case .maximum:
            pool = [Float](repeating: -Float.infinity, count: totalSize)
            inputUsed = [Int](repeating: 0, count: totalSize)
        case .average:
            pool = [Float](repeating: 0.0, count: totalSize)
        }
        
        //  Reduce each dimension
        for w in 0..<resultSize[3] {
            let wResultStart = w * resultStride[3]
            for wGroup in 0..<reduction[3] {
                let wSourceStart = ((w*reduction[3] + wGroup) * sourceStride[3])
                for z in 0..<resultSize[2] {
                    let zResultStart = wResultStart + (z * resultStride[2])
                    for zGroup in 0..<reduction[2] {
                        let zSourceStart = wSourceStart + ((z*reduction[2] + zGroup) * sourceStride[2])
                        for y in 0..<resultSize[1] {
                            let yResultStart = zResultStart + (y * resultStride[1])
                            for yGroup in 0..<reduction[1] {
                                let ySourceStart = zSourceStart + ((y*reduction[1] + yGroup) * sourceStride[1])
                                for x in 0..<resultSize[0] {
                                    let resultIndex = yResultStart + x
                                    let xSourceStart = ySourceStart + x*reduction[0]
                                    for xGroup in 0..<reduction[0] {
                                        let sourceIndex = xSourceStart + xGroup
                                        switch poolType {
                                        case .minimum:
                                            if (inputs[sourceIndex] < pool[resultIndex]) {
                                                pool[resultIndex] = inputs[sourceIndex]
                                                inputUsed![resultIndex] = sourceIndex
                                            }
                                        case .maximum:
                                            if (inputs[sourceIndex] > pool[resultIndex]) {
                                                pool[resultIndex] = inputs[sourceIndex]
                                                inputUsed![resultIndex] = sourceIndex
                                            }
                                        case .average:
                                            pool[resultIndex] += inputs[sourceIndex]
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        if (poolType == .average) {
            var totalCells = 1
            for index in 0..<4 { totalCells *= reduction[index] }
            var multiplier : Float = 1.0 / Float(totalCells)
            vDSP_vsmul(pool, 1, &multiplier, &pool, 1, vDSP_Length(pool.count))
        }
        
        return pool
    }
    
    public func getResults() -> [Float]
    {
        return pool
    }
    
    public func getResultSize() -> DeepChannelSize
    {
        return resultSize
    }
    
    public func getResultRange() ->(minimum: Float, maximum: Float)
    {
        //  Result range is a function of the input range.  Default to a value that will work, even if not optimum
        return (minimum: 0.0, maximum: 1.0)
    }
    
    public func startBatch()
    {
        //  No weights in a pooling operation
    }
    
    //  𝟃E/𝟃h comes in, 𝟃E/𝟃x goes out
    public func backPropogateGradient(_ upStreamGradient: [Float]) -> [Float]
    {
        var downstreamGradient = [Float](repeating: 0.0, count: inputSize.totalSize)
        
        //  If this is a minimum or maximum pooling, we have the use information to set the downstream gradient
        if (poolType != .average) {
            for sourceIndex in 0..<inputUsed!.count {
                downstreamGradient[inputUsed![sourceIndex]] = upStreamGradient[sourceIndex]
            }
            return downstreamGradient
        }
    
        //  If average, the gradient just gets spread over the input space that was pooled
        //  Get spread factor for each dimension
        var sourceSize = resultSize.dimensions
        sourceSize += [1, 1, 1]     //  Add size for missing dimensions
        var destSize = inputSize.dimensions
        destSize += [1, 1, 1]     //  Add size for missing dimensions
        let spreadW = destSize[3] / sourceSize[3]
        let spreadZ = destSize[2] / sourceSize[2]
        let spreadY = destSize[1] / sourceSize[1]
        let spreadX = destSize[0] / sourceSize[0]
        let multiplier = 1.0 / Float(spreadW * spreadZ * spreadY * spreadX)
        
        //  Determine the stride for each dimension
        var sourceStride = [Int](repeating: 1, count: 4)
        var destStride = [Int](repeating: 1, count: 4)
        for index in 1..<4 {
            for i in 0..<index { sourceStride[index] *= sourceSize[i] }
            for i in 0..<index { destStride[index] *= destSize[i] }
        }
        
        //  Spread each dimension
        var destIndex = 0
        var wSourceStart = 0
        for _ in 0..<sourceSize[3] {                            //  Each source W
            for _ in 0..<spreadW {                              //  Spread to each destination W
                var zSourceStart = wSourceStart
                for _ in 0..<sourceSize[2] {                    //  Each source Z
                    for _ in 0..<spreadZ {                      //  Spread to each destination Z
                        var ySourceStart = zSourceStart
                        for _ in 0..<sourceSize[1] {            //  Each source Y
                            for _ in 0..<spreadY {              //  Spread to each destination Y
                                for x in 0..<sourceSize[0] {    //  Each source X
                                    let sourceIndex = ySourceStart + x
                                    for _ in 0..<spreadX {      //  Spread to each destination X
                                        downstreamGradient[destIndex] = upStreamGradient[sourceIndex] * multiplier
                                        destIndex += 1
                                    }
                                }
                            }
                            ySourceStart += sourceStride[1]
                        }
                    }
                    zSourceStart += sourceStride[2]
                }
            }
            wSourceStart += sourceStride[3]
        }
        
        return downstreamGradient
    }
    
    public func updateWeights(_ trainingRate : Float, weightDecay: Float)
    {
        //  No weights in a pooling operation
    }
    
    public func gradientCheck(ε: Float, Δ: Float, network: DeepNetwork) -> Bool
    {
        //  No (stored) gradients in a pooling layer
        return true
    }
    
    public func getPersistenceDictionary() -> [String: AnyObject]
    {
        var resultDictionary : [String: AnyObject] = [:]
        
        //  Set the pooling type
        resultDictionary["poolingType"] = poolType.rawValue as AnyObject?
        
        //  Set the dimension
        resultDictionary["dimension"] = dimension as AnyObject?
        
        //  Set the reduction levels
        resultDictionary["reductionLevels"] = reductionLevels as AnyObject?
        
        return resultDictionary
    }
}
