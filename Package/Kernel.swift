//
//  Kernel.swift
//  AIToolbox
//
//  Created by Kevin Coble on 12/6/15.
//  Copyright © 2015 Kevin Coble. All rights reserved.
//

import Foundation
#if os(Linux)
#else
import Accelerate
#endif

public enum SVMKernelType   //  SVM kernel type
{
    case linear
    case polynomial
    case radialBasisFunction
    case sigmoid
    case precomputed
}

public struct KernelParameters {
    let type: SVMKernelType
    let degree: Int         //  for polynomial
    let gamma: Double       //  for polynomial, radialbasis, sigmoid
    let coef0: Double       //  for polynomial, sigmoid
    
    public init(type: SVMKernelType, degree: Int, gamma: Double, coef0: Double) {
        self.type = type
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
    }

}

class Kernel {
    
    //  Problem data
    var problemData : MLCombinedDataSet
    
    //  Diagonal items squared (for RBF)
    let x_square: [Double]!
    
    let kernelType: SVMKernelType
    //  Closure with kernel function - initialize with lazy-var to avoid 'Variable kernel_function used before initialized' error when just a 'let'
    lazy var kernel_function : (Int, Int) -> Double = self.initKernelFunction()
    lazy var QDiagonal : [Double] = self.initQDiagonal()
    let degree: Int
    let gamma: Double
    let coef0: Double
    
    init(parameters: KernelParameters, data: MLCombinedDataSet)
    {
        kernelType = parameters.type
        problemData = data
        self.degree = parameters.degree
        self.gamma = parameters.gamma
        self.coef0 = parameters.coef0
        x_square = []
        
        if (parameters.type == .radialBasisFunction) {
            for i in 0..<data.size {
                x_square.append(dotProduct(i,i))
            }
        }
    }
    
    func initKernelFunction() -> ((Int, Int) -> Double) {
        switch (kernelType) {
        case .linear:
            return dotProduct
            
        case .polynomial:
            return polyKernel
            
        case .radialBasisFunction:
            return RBFKernel
            
        case .sigmoid:
            return sigmoidKernel
            
        case .precomputed:
            return precomputedKernel
        }
    }
    
    func initQDiagonal() -> [Double] {
        var returnArray: [Double] = []
        
        for i in 0..<problemData.size {
            returnArray.append(kernel_function(i,i))
        }
        
        return returnArray
    }
    
    func getQ(_ i: Int) ->[Double]
    {
        return []
    }
    
    func getQDiagonal() ->[Double]
    {
        return QDiagonal
    }
    
    class func calcKernelValue(_ parameters: KernelParameters, x: [Double], y: [Double]) ->Double
    {
        switch (parameters.type) {
        case .linear:
            var sum = 0.0
            vDSP_dotprD(x, 1, y, 1, &sum, vDSP_Length(x.count))
            return sum
            
        case .polynomial:
            var sum = 0.0
            vDSP_dotprD(x, 1, y, 1, &sum, vDSP_Length(x.count))
            var tmp = parameters.gamma * sum + parameters.coef0
            var ret = 1.0
            
            var t = parameters.degree
            while (t > 0) {
                if (t%2==1) { ret *= tmp }
                tmp = tmp * tmp
                t /= 2
            }
            return ret
            
        case .radialBasisFunction:
            var diff = [Double](repeating: 0.0, count: x.count)
            vDSP_vsubD(x, 1, y, 1, &diff, 1, vDSP_Length(x.count))
            var sum = 0.0
            vDSP_dotprD(diff, 1, diff, 1, &sum, vDSP_Length(x.count))
            
            return exp(-parameters.gamma * sum)
            
        case .sigmoid:
            var sum = 0.0
            vDSP_dotprD(x, 1, y, 1, &sum, vDSP_Length(x.count))
            return tanh(parameters.gamma * sum + parameters.coef0)
            
        case .precomputed:
            //!!  not yet implemented
            return 0.0
        }
    }
    
    func dotProduct(_ vector1Index : Int, _ vector2Index : Int) -> Double
    {
        var sum = 0.0
        do {
            let vector1 = try problemData.getInput(vector1Index)
            let vector2 = try problemData.getInput(vector2Index)
            vDSP_dotprD(vector1, 1, vector2, 1, &sum, vDSP_Length(problemData.inputDimension))
        }
        catch {
            print("invalid index in kernel dotProduct - \(vector1Index) or \(vector2Index)")
        }
        return sum
    }
    
    func polyKernel(_ vector1Index : Int, _ vector2Index : Int) -> Double
    {
        var tmp = gamma * dotProduct(vector1Index, vector2Index) + coef0
        var ret = 1.0
        
        var t = degree
        while (t > 0) {
            if (t%2==1) { ret *= tmp }
            tmp = tmp * tmp
            t /= 2
        }
        return ret
    }
    
    func RBFKernel(_ vector1Index : Int, _ vector2Index : Int) -> Double
    {
        return exp(-gamma*(x_square[vector1Index]+x_square[vector2Index] - 2 * dotProduct(vector1Index, vector2Index)))
    }
    
    func sigmoidKernel(_ vector1Index : Int, _ vector2Index : Int) -> Double
    {
        return tanh(gamma * dotProduct(vector1Index, vector2Index) + coef0)
    }
    
    func precomputedKernel(_ vector1Index : Int, _ vector2Index : Int) -> Double
    {
        //!!  not yet implemented
        return 0.0
    }
}


class SVCKernel : Kernel {
    let outputs : [Double]
    
    init(parameters: KernelParameters, data: MLCombinedDataSet, outputs: [Double])
    {
        self.outputs = outputs
        
        super.init(parameters: parameters, data: data)
    }
    
    override func getQ(_ i: Int) ->[Double]
    {
        var data: [Double] = []
        for j in 0..<problemData.size {
            data.append(outputs[i] * outputs[j] * kernel_function(i,j))
        }
        
        return data
    }
}

class OneClassKernel : Kernel {
    
    override func getQ(_ i: Int) ->[Double]
    {
        var data: [Double] = []
        for j in 0..<problemData.size {
            data.append(kernel_function(i,j))
        }
        
        return data
    }
}

class SVRKernel : Kernel {
    
    var sign : [Double]
    
    override init(parameters: KernelParameters, data: MLCombinedDataSet) {
        sign = [Double](repeating: 1.0, count: data.size)
        sign += [Double](repeating: -1.0, count: data.size)
        
        super.init(parameters: parameters, data: data)
    }
    
    override func initQDiagonal() -> [Double] {
        var returnArray: [Double] = []
        
        for i in 0..<problemData.size {
            returnArray.append(kernel_function(i,i))
        }
        
        return returnArray + returnArray
    }
    
    override func getQ(_ i: Int) ->[Double]
    {
        var real_i = i
        if (real_i >= problemData.size) {real_i -= problemData.size}
        var data: [Double] = []
        let signI = sign[i]
        for j in 0..<(problemData.size*2) {
            var real_j = j
            if (real_j >= problemData.size) {real_j -= problemData.size}
            data.append(signI * sign[j] * kernel_function(real_i, real_j))
        }
        
        return data
    }
}
