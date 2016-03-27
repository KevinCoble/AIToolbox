//
//  Kernel.swift
//  AIToolbox
//
//  Created by Kevin Coble on 12/6/15.
//  Copyright © 2015 Kevin Coble. All rights reserved.
//

import Foundation
import Accelerate

public enum SVMKernalType   //  SVM kernal type
{
    case Linear
    case Polynomial
    case RadialBasisFunction
    case Sigmoid
    case Precomputed
}

public struct KernelParameters {
    let type: SVMKernalType
    let degree: Int         //  for polynomial
    let gamma: Double       //  for polynomial, radialbasis, sigmoid
    let coef0: Double       //  for polynomial, sigmoid
    
    public init(type: SVMKernalType, degree: Int, gamma: Double, coef0: Double) {
        self.type = type
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
    }

}

class Kernel {
    
    //  Problem data
    var problemData : DataSet
    
    //  Diagonal items squared (for RBF)
    let x_square: [Double]!
    
    let kernalType: SVMKernalType
    //  Closure with kernal function - initialize with lazy-var to avoid 'Variable kernal_function used before initialized' error when just a 'let'
    lazy var kernal_function : (Int, Int) -> Double = self.initKernalFunction()
    lazy var QDiagonal : [Double] = self.initQDiagonal()
    let degree: Int
    let gamma: Double
    let coef0: Double
    
    init(parameters: KernelParameters, data: DataSet)
    {
        kernalType = parameters.type
        problemData = data
        self.degree = parameters.degree
        self.gamma = parameters.gamma
        self.coef0 = parameters.coef0
        x_square = []
        
        if (parameters.type == .RadialBasisFunction) {
            for i in 0..<data.size {
                x_square.append(dotProduct(i,i))
            }
        }
    }
    
    func initKernalFunction() -> ((Int, Int) -> Double) {
        switch (kernalType) {
        case .Linear:
            return dotProduct
            
        case .Polynomial:
            return polyKernel
            
        case .RadialBasisFunction:
            return RBFKernel
            
        case .Sigmoid:
            return sigmoidKernel
            
        case .Precomputed:
            return precomputedKernel
        }
    }
    
    func initQDiagonal() -> [Double] {
        var returnArray: [Double] = []
        
        for i in 0..<problemData.size {
            returnArray.append(kernal_function(i,i))
        }
        
        return returnArray
    }
    
    func getQ(i: Int) ->[Double]
    {
        return []
    }
    
    func getQDiagonal() ->[Double]
    {
        return QDiagonal
    }
    
    class func calcKernelValue(parameters: KernelParameters, x: [Double], y: [Double]) ->Double
    {
        switch (parameters.type) {
        case .Linear:
            var sum = 0.0
            vDSP_dotprD(x, 1, y, 1, &sum, vDSP_Length(x.count))
            return sum
            
        case .Polynomial:
            var sum = 0.0
            vDSP_dotprD(x, 1, y, 1, &sum, vDSP_Length(x.count))
            var tmp = parameters.gamma * sum + parameters.coef0
            var ret = 1.0
            
            for var t = parameters.degree; t > 0; t/=2 {
                if (t%2==1) { ret *= tmp }
                tmp = tmp * tmp
            }
            return ret
            
        case .RadialBasisFunction:
            var diff = [Double](count: x.count, repeatedValue: 0.0)
            vDSP_vsubD(x, 1, y, 1, &diff, 1, vDSP_Length(x.count))
            var sum = 0.0
            vDSP_dotprD(diff, 1, diff, 1, &sum, vDSP_Length(x.count))
            
            return exp(-parameters.gamma * sum)
            
        case .Sigmoid:
            var sum = 0.0
            vDSP_dotprD(x, 1, y, 1, &sum, vDSP_Length(x.count))
            return tanh(parameters.gamma * sum + parameters.coef0)
            
        case .Precomputed:
            //!!  not yet implemented
            return 0.0
        }
    }
    
    func dotProduct(vector1Index : Int, _ vector2Index : Int) -> Double
    {
        var sum = 0.0
        vDSP_dotprD(problemData.inputs[vector1Index], 1, problemData.inputs[vector2Index], 1, &sum, vDSP_Length(problemData.inputDimension))
        
        return sum
    }
    
    func polyKernel(vector1Index : Int, _ vector2Index : Int) -> Double
    {
        var tmp = gamma * dotProduct(vector1Index, vector2Index) + coef0
        var ret = 1.0
        
        for var t = degree; t > 0; t/=2 {
            if (t%2==1) { ret *= tmp }
            tmp = tmp * tmp
        }
        return ret
    }
    
    func RBFKernel(vector1Index : Int, _ vector2Index : Int) -> Double
    {
        return exp(-gamma*(x_square[vector1Index]+x_square[vector2Index] - 2 * dotProduct(vector1Index, vector2Index)))
    }
    
    func sigmoidKernel(vector1Index : Int, _ vector2Index : Int) -> Double
    {
        return tanh(gamma * dotProduct(vector1Index, vector2Index) + coef0)
    }
    
    func precomputedKernel(vector1Index : Int, _ vector2Index : Int) -> Double
    {
        //!!  not yet implemented
        return 0.0
    }
}


class SVCKernel : Kernel {
    let outputs : [Double]
    
    init(parameters: KernelParameters, data: DataSet, outputs: [Double])
    {
        self.outputs = outputs
        
        super.init(parameters: parameters, data: data)
    }
    
    override func getQ(i: Int) ->[Double]
    {
        var data: [Double] = []
        for j in 0..<problemData.size {
            data.append(outputs[i] * outputs[j] * kernal_function(i,j))
        }
        
        return data
    }
}

class OneClassKernel : Kernel {
    
    override func getQ(i: Int) ->[Double]
    {
        var data: [Double] = []
        for j in 0..<problemData.size {
            data.append(kernal_function(i,j))
        }
        
        return data
    }
}

class SVRKernel : Kernel {
    
    var sign : [Double]
    
    override init(parameters: KernelParameters, data: DataSet) {
        sign = [Double](count: data.size, repeatedValue: 1.0)
        sign += [Double](count: data.size, repeatedValue: -1.0)
        
        super.init(parameters: parameters, data: data)
    }
    
    override func initQDiagonal() -> [Double] {
        var returnArray: [Double] = []
        
        for i in 0..<problemData.size {
            returnArray.append(kernal_function(i,i))
        }
        
        return returnArray + returnArray
    }
    
    override func getQ(i: Int) ->[Double]
    {
        var real_i = i
        if (real_i >= problemData.size) {real_i -= problemData.size}
        var data: [Double] = []
        let signI = sign[i]
        for j in 0..<(problemData.size*2) {
            var real_j = j
            if (real_j >= problemData.size) {real_j -= problemData.size}
            data.append(signI * sign[j] * kernal_function(real_i, real_j))
        }
        
        return data
    }
}