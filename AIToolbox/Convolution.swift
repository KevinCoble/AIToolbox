//
//  Convolution.swift
//  AIToolbox
//
//  Created by Kevin Coble on 2/13/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation
import Accelerate

public enum Convolution2DMatrix : Int
{
    case verticalEdge3 = 0
    case horizontalEdge3
    case custom3
    case learnable3
    
    public func getString() ->String
    {
        switch self {
        case .verticalEdge3:
            return "Vertical Edge 3x3"
        case .horizontalEdge3:
            return "Horizontal Edge 3x3"
        case .custom3:
            return "Custom 3x3"
        case .learnable3:
            return "Learnable 3x3"
        }
    }
    
    public func getDefaultMatrix() ->[Float]
    {
        var matrix : [Float]
        switch self {
        case .verticalEdge3:
            matrix = [-1, 0, 1, -2, 0, 2, -1, 0 , 1]
        case .horizontalEdge3:
            matrix =  [-1, -2, -1, 0, 0, 0, 1, 2 , 1]
        case .custom3:
            matrix =  [0, 0, 0, 0, 1, 0, 0, 0, 0]      //  Default to the identity convolution
        case .learnable3:
            matrix =  [0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111]      //  Default to the average convolution
        }
        
        return matrix
    }
    
    public func getMatrixSize() ->Int
    {
        switch self {
        case .verticalEdge3, .horizontalEdge3, .custom3, .learnable3:
            return 3
        }
    }
    
    public func getCustomOfSameSize() ->Convolution2DMatrix
    {
        switch self {
        case .verticalEdge3, .horizontalEdge3, .custom3, .learnable3:
            return .custom3
        }
    }
}

final public class Convolution2D : DeepNetworkOperator
{
    public private(set) var matrixType : Convolution2DMatrix
    public private(set) var matrix : [Float] {
        didSet {
            determineResultRange()
        }
    }
    var minResult : Float = -1.0
    var maxResult : Float = 1.0
    var convolution : [Float] = []
    var resultSize : DeepChannelSize
    var weightAccumulations : [Float] = []
    fileprivate var lastInputs : [Float] = []       //  Last inputs used

    public init(usingMatrix : Convolution2DMatrix)
    {
        matrixType = usingMatrix
        matrix = matrixType.getDefaultMatrix()
        resultSize = DeepChannelSize(dimensionCount: 0, dimensionValues: [])
        determineResultRange()
    }
    
    public init?(fromDictionary: [String: AnyObject])
    {
        //  Init for nil return (hopefully Swift 3 removes this need)
        resultSize = DeepChannelSize(dimensionCount: 0, dimensionValues: [])
        matrix = []

        //  Get the matrix type
        let matrixTypeValue = fromDictionary["matrixType"] as? NSInteger
        if matrixTypeValue == nil { return nil }
        let tempMatrixType = Convolution2DMatrix(rawValue: matrixTypeValue!)
        if (tempMatrixType == nil) { return nil }
        matrixType = tempMatrixType!
        
        //  Get the matrix
        let tempArray = getFloatArray(fromDictionary, identifier: "matrix")
        if (tempArray == nil) { return nil }
        matrix = tempArray!
        determineResultRange()
    }
    
    public func setMatrixType(type : Convolution2DMatrix)
    {
        matrixType = type
        matrix = type.getDefaultMatrix()
    }
    
    public func setMatrixValue(atIndex: Int, toValue: Float)
    {
        matrix[atIndex] = toValue
    }

    public func determineResultRange()
    {
        minResult = 0.0
        maxResult = 0.0
        for element in matrix {
            if element < 0 {
                minResult += element
            }
            else {
                maxResult += element
            }
        }
    }
    
    public func getType() -> DeepNetworkOperatorType
    {
        return .convolution2DOperation
    }
    
    public func getDetails() -> String
    {
        return matrixType.getString()
    }

    public func getResultingSize(_ inputSize: DeepChannelSize) -> DeepChannelSize
    {
        //  A convolution doesn't change the size
        resultSize = inputSize
        return inputSize
    }
    
    public func initializeParameters()
    {
        //  Initialize the parameters if we are a learning convolution
        if (matrixType == .learnable3) {
            let weightDiviser : Float = 1.0 / sqrt(9)       //  Xavier initialization
            for i in 0..<9 {
                matrix[i] = Gaussian.gaussianRandomFloat(0.0, standardDeviation : 1.0) * weightDiviser
            }
        }
    }
    
    public func feedForward(_ inputs: [Float], inputSize: DeepChannelSize) -> [Float]
    {
        lastInputs = inputs
        let matrixSize = UInt32(matrixType.getMatrixSize())

        //  Get the source data as a vImage buffer
        var source: vImage_Buffer = withUnsafePointer(to: &lastInputs) {
            return vImage_Buffer(data: UnsafeMutableRawPointer(mutating: $0), height: vImagePixelCount(inputSize.dimensions[1]), width: vImagePixelCount(inputSize.dimensions[0]), rowBytes: inputSize.dimensions[0] * MemoryLayout<Float>.size)
        }

        //  Create a destination as a vImage buffer
        convolution = [Float](repeating: 0.0, count: inputs.count)
        var dest = withUnsafePointer(to: &convolution) {
            return vImage_Buffer(data: UnsafeMutableRawPointer(mutating: $0), height: vImagePixelCount(inputSize.dimensions[1]), width: vImagePixelCount(inputSize.dimensions[0]), rowBytes: inputSize.dimensions[0] * MemoryLayout<Float>.size)
        }

        //  Convolve
        let error = vImageConvolve_PlanarF(&source, &dest, nil, 0, 0, matrix, matrixSize, matrixSize, 0.0, UInt32(kvImageEdgeExtend))
        if (error != kvImageNoError) {
            convolution = []
        }
        
        return convolution
    }
    
    public func getResults() -> [Float]
    {
        return convolution
    }

    public func getResultSize() -> DeepChannelSize
    {
        return resultSize
    }
    
    
    public func getResultRange() ->(minimum: Float, maximum: Float)
    {
        return (minimum: minResult, maximum: maxResult)
    }
    
    public func startBatch()
    {
        if (matrixType != .learnable3) { return }
        
        weightAccumulations = [Float](repeating: 0.0, count: matrix.count)
    }
    
    //  𝟃E/𝟃h comes in, 𝟃E/𝟃x goes out
    public func backPropogateGradient(_ upStreamGradient: [Float]) -> [Float]
    {
        if (matrixType != .learnable3) { return upStreamGradient }      //  If not learnable, just pass the gradient downstream
        
        //  Allocate the downstream gradient
        var downStreamGradient = [Float](repeating: 0.0, count: resultSize.totalSize)
        
        //  Get the convolution index offset
        let convolutionSize = matrixType.getMatrixSize()
        let offset = -convolutionSize / 2       //  Size will be odd, so this puts the middle at 0
        
        //  Iterate through each row of the upstream gradient
        let numRows = resultSize.dimensions[1]
        let numColumns = resultSize.dimensions[0]
        for row in 0..<numRows {
            for convRow in 0..<convolutionSize {
                var sourceRow = row + convRow + offset
                //        if (sourceRow < 0 || sourceRow >= numRows) { continue }   //  If kv​Image​Truncate​Kernel used in convolution
                if (sourceRow < 0) { sourceRow = 0 }                        //  If kvImageEdgeExtend used in convolution
                if (sourceRow >= numRows) { sourceRow = numRows - 1 }       //  If kvImageEdgeExtend used in convolution
                let sourceIndex = sourceRow * numColumns
                var destIndex = row * numColumns
                let convolutionIndex = convRow * convolutionSize
                for column in 0..<numColumns {
                    for convColumn in 0..<convolutionSize {
                        var sourceColumn = column + convColumn + offset
                        //                if (sourceColumn < 0 || sourceColumn >= numColumns) { continue }  //  If kv​Image​Truncate​Kernel used in convolution
                        if (sourceColumn < 0) { sourceColumn = 0 }                          //  If kvImageEdgeExtend used in convolution
                        if (sourceColumn >= numColumns) { sourceColumn = numColumns - 1 }   //  If kvImageEdgeExtend used in convolution
                        let sourceLocation = sourceIndex + sourceColumn
                        weightAccumulations[convolutionIndex+convColumn] += upStreamGradient[destIndex] * lastInputs[sourceLocation]
                        downStreamGradient[sourceLocation] += upStreamGradient[destIndex] * matrix[convolutionIndex+convColumn]
                    }
                    destIndex += 1
                }
            }
            
        }
        
        return downStreamGradient
    }
    
    public func updateWeights(_ trainingRate : Float, weightDecay: Float)
    {
        //  Only update if learnable
        if (matrixType != .learnable3) { return }
        
        //  If there is a decay factor, use it
        if (weightDecay != 1.0) {
            var λ = weightDecay     //  Needed for unsafe pointer conversion
            vDSP_vsmul(matrix, 1, &λ, &matrix, 1, vDSP_Length(matrix.count))
        }
        
        //  Subtract the weight changes from the weight matrix (W = W - η∇)
        var η = -trainingRate     //  Needed for unsafe pointer conversion
        vDSP_vsma(weightAccumulations, 1, &η, matrix, 1, &matrix, 1, vDSP_Length(matrix.count))
    }
    
    public func gradientCheck(ε: Float, Δ: Float, network: DeepNetwork) -> Bool
    {
        //  Only check if learnable
        if (matrixType != .learnable3) { return true}
        var result = true
        
        //  Iterate through each parameter
        for index in 0..<matrix.count {
            let oldValue = matrix[index]
            
            //  Get the network results with a small addition to the parameter
            matrix[index] += ε
            network.feedForward()
            let plusLoss = network.getResultLoss()
            
            //  Get the network results with a small subtraction from the parameter
            matrix[index] = oldValue - ε
            network.feedForward()
            let minusLoss = network.getResultLoss()
            matrix[index] = oldValue
            
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
        
        //  Set the matrix type
        resultDictionary["matrixType"] = matrixType.rawValue as AnyObject?
        
        //  Set the matrix
        resultDictionary["matrix"] = matrix as AnyObject?
        
        return resultDictionary
    }
}
