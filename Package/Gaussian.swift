//
//  Gaussian.swift
//  AIToolbox
//
//  Created by Kevin Coble on 4/11/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation
#if os(Linux)
import Glibc
#else
import Accelerate
#endif


public enum GaussianError: Error {
    case dimensionError
    case zeroInVariance
    case inverseError
    case badVarianceValue
    case diagonalCovarianceOnly
    case errorInSVDParameters
    case svdDidNotConverge
}

open class Gaussian {
    //  Parameters
    var σsquared : Double
    var mean : Double
    var multiplier : Double
    
    
    ///  Create a gaussian
    public init(mean: Double, variance: Double) throws {
        if (variance < 0.0) { throw GaussianError.badVarianceValue }
        self.mean = mean
        σsquared = variance
        multiplier = 1.0 / sqrt(σsquared * 2.0 * Double.pi)
    }
    
    open func setMean(_ mean: Double)
    {
        self.mean = mean
    }
    
    open func setVariance(_ variance: Double)
    {
        σsquared = variance
        multiplier = 1.0 / sqrt(σsquared * 2.0 * Double.pi)
    }
    
    open func setParameters(_ parameters: [Double]) throws
    {
        if (parameters.count < 2) { throw MachineLearningError.notEnoughData }
        
        mean = parameters[0]
        setVariance(parameters[1])
        
    }
    open func getParameterDimension() -> Int
    {
        return 2  //  Mean and variance
    }
    open func getParameters() throws -> [Double]
    {
        var parameters = [mean]
        parameters.append(σsquared)
        return parameters
    }
    
    ///  Function to get the probability of an input value
    open func getProbability(_ input: Double) -> Double {
        let exponent = (input - mean) * (input - mean) / (-2.0 * σsquared)
        return multiplier * exp(exponent)
    }
    
    ///  Function to get a random value
    open func gaussRandom() -> Double {
        return Gaussian.gaussianRandom(mean, standardDeviation: sqrt(σsquared))
    }
    
    static var y2 = 0.0
    static var use_last = false
    ///  static Function to get a random value for a given distribution
    open static func gaussianRandom(_ mean : Double, standardDeviation : Double) -> Double
    {
        var y1 : Double
        if (use_last)		        /* use value from previous call */
        {
            y1 = y2
            use_last = false
        }
        else
        {
            var w = 1.0
            var x1 = 0.0
            var x2 = 0.0
            repeat {
#if os(Linux)
                x1 = 2.0 * (Double(random()) / Double(RAND_MAX)) - 1.0
                x2 = 2.0 * (Double(random()) / Double(RAND_MAX)) - 1.0
#else
                x1 = 2.0 * (Double(arc4random()) / Double(UInt32.max)) - 1.0
                x2 = 2.0 * (Double(arc4random()) / Double(UInt32.max)) - 1.0
#endif
                w = x1 * x1 + x2 * x2
            } while ( w >= 1.0 )
            
            w = sqrt( (-2.0 * log( w ) ) / w )
            y1 = x1 * w
            y2 = x2 * w
            use_last = true
        }
        
        return( mean + y1 * standardDeviation )
    }
    
    static var y2Float: Float = 0.0
    static var use_lastFloat = false
    ///  static Function to get a random value for a given distribution
    open static func gaussianRandomFloat(_ mean : Float, standardDeviation : Float) -> Float
    {
        var y1 : Float
        if (use_last)		        /* use value from previous call */
        {
            y1 = y2Float
            use_last = false
        }
        else
        {
            var w : Float = 1.0
            var x1 : Float = 0.0
            var x2 : Float = 0.0
            repeat {
#if os(Linux)
                x1 = 2.0 * (Float(random()) / Float(RAND_MAX)) - 1.0
                x2 = 2.0 * (Float(random()) / Float(RAND_MAX)) - 1.0
#else
                x1 = 2.0 * (Float(arc4random()) / Float(UInt32.max)) - 1.0
                x2 = 2.0 * (Float(arc4random()) / Float(UInt32.max)) - 1.0
#endif
                w = x1 * x1 + x2 * x2
            } while ( w >= 1.0 )
            
            w = sqrt( (-2.0 * log( w ) ) / w )
            y1 = x1 * w
            y2Float = x2 * w
            use_last = true
        }
        
        return( mean + y1 * standardDeviation )
    }
}

#if os(Linux)
#else
open class MultivariateGaussian {
    
    //  Parameters
    let dimension: Int
    let diagonalΣ : Bool
    var μ : [Double]    //  Mean
    var Σ : [Double]    //  Covariance.  If diagonal, then vector, else column-major square matrix (column major for LAPACK)
    
    //  Calculate values for computing probability
    var haveCalcValues = false
    var multiplier : Double       //  The 1/(2π) ^ (dimension / 2) sqrt(detΣ)
    var invΣ : [Double]     //  Inverse of Σ (1/Σ if diagonal)
    
    
    ///  Create a multivariate gaussian.  dimension should be 2 or greater
    public init(dimension: Int, diagonalCovariance: Bool = true) throws {
        self.dimension = dimension
        diagonalΣ = diagonalCovariance
        if (dimension < 2) { throw GaussianError.dimensionError }
        
        //  Start with 0 mean
        μ = [Double](repeating: 0.0, count: dimension)
        
        //  Start with the identity matrix for covariance
        if (diagonalΣ) {
            Σ = [Double](repeating: 1.0, count: dimension)
            invΣ = [Double](repeating: 1.0, count: dimension)
        }
        else {
            Σ = [Double](repeating: 0.0, count: dimension * dimension)
            for index in 0..<dimension { Σ[index * dimension + index] = 1.0 }
            invΣ = [Double](repeating: 0.0, count: dimension * dimension)       //  Will get calculated later
        }
        
        //  Set the multiplier temporarily
        multiplier = 1.0
    }
    
    fileprivate func getComputeValues() throws {
        var denominator = pow(2.0 * Double.pi, Double(dimension) * 0.5)
        
        //  Get the determinant and inverse of the covariance matrix
        var sqrtDeterminant = 1.0
        if (diagonalΣ) {
            for index in 0..<dimension {
                sqrtDeterminant *= Σ[index]
                invΣ[index] = 1.0 / Σ[index]
            }
            sqrtDeterminant = sqrt(sqrtDeterminant)
        }
        else {
            let uploChar = "U" as NSString
            var uplo : Int8 = Int8(uploChar.character(at: 0))          //  use upper triangle
            var A = Σ       //  Make a copy so it isn't mangled
            var n : __CLPK_integer = __CLPK_integer(dimension)
            var info : __CLPK_integer = 0
            dpotrf_(&uplo, &n, &A, &n, &info)
            if (info != 0) { throw GaussianError.inverseError }
            //  Extract sqrtDeterminant from U by multiplying the diagonal  (U is multiplied by Utranspose after factorization)
            for index in 0..<dimension {
                sqrtDeterminant *= A[index * dimension + index]
            }
            
            //  Get the inverse
            dpotri_(&uplo, &n, &A, &n, &info)
            if (info != 0) { throw GaussianError.inverseError }
            
            //  Convert inverse U into symmetric full matrix for matrix multiply routines
            for row in 0..<dimension {
                for column in row..<dimension {
                    invΣ[row * dimension + column] = A[column * dimension + row]
                    invΣ[column * dimension + row] = A[column * dimension + row]
                }
           }
        }
        
        denominator *= sqrtDeterminant
        
        if (denominator == 0.0) { throw GaussianError.zeroInVariance }
        multiplier = 1.0 / denominator
        
        haveCalcValues = true
    }
    
    ///  Function to set the mean
    open func setMean(_ mean: [Double]) throws {
        if (mean.count != dimension) { throw GaussianError.dimensionError }
        μ = mean
    }
    
    
    ///  Function to set the covariance values.  Values are copied into symmetric sides of matrix
    open func setCoVariance(_ inputIndex1: Int, inputIndex2: Int, value: Double) throws {
        if (value < 0.0) { throw GaussianError.badVarianceValue }
        if (inputIndex1 < 0 || inputIndex1 >= dimension) { throw GaussianError.badVarianceValue }
        if (inputIndex2 < 0 || inputIndex2 >= dimension) { throw GaussianError.badVarianceValue }
        if (diagonalΣ && inputIndex1 != inputIndex2) { throw GaussianError.diagonalCovarianceOnly }
        
        Σ[inputIndex1 * dimension + inputIndex2] = value
        Σ[inputIndex2 * dimension + inputIndex1] = value
        
        haveCalcValues = false
    }
    
    open func setCovarianceMatrix(_ matrix: [Double]) throws {
        if (diagonalΣ && matrix.count != dimension) { throw GaussianError.diagonalCovarianceOnly }
        if (!diagonalΣ && matrix.count != dimension * dimension) { throw GaussianError.dimensionError }
        Σ = matrix
        haveCalcValues = false
    }
    
    open func setParameters(_ parameters: [Double]) throws
    {
        let requiredSize = getParameterDimension()
        if (parameters.count < requiredSize) { throw MachineLearningError.notEnoughData }
        
        μ = Array(parameters[0..<dimension])
        try setCovarianceMatrix(Array(parameters[dimension..<requiredSize]))
        
    }
    open func getParameterDimension() -> Int
    {
        var numParameters = dimension   //  size of the mean
        if (diagonalΣ) {
            numParameters += dimension      //  size of diagonal covariance matrix
        }
        else {
            numParameters += dimension * dimension      //  size of full covariance matrix
        }
        
        return numParameters
    }
    open func getParameters() throws -> [Double]
    {
        var parameters = μ
        parameters += Σ
        return parameters
    }
    
    ///  Function to get the probability of an input vector
    open func getProbability(_ inputs: [Double]) throws -> Double {
        if (inputs.count != dimension) { throw GaussianError.dimensionError }
        if (!haveCalcValues) {
            do {
                try getComputeValues()
            }
            catch let error {
                throw error
            }
        }
        
        //  Subtract the mean
        var relative = [Double](repeating: 0.0, count: dimension)
        vDSP_vsubD(μ, 1, inputs, 1, &relative, 1, vDSP_Length(dimension))
        
        //  Determine the exponent
        var partial = [Double](repeating: 0.0, count: dimension)
        if (diagonalΣ) {
            vDSP_vmulD(relative, 1, invΣ, 1, &partial, 1, vDSP_Length(dimension))
        }
        else {
            vDSP_mmulD(invΣ, 1, relative, 1, &partial, 1, vDSP_Length(dimension), vDSP_Length(1), vDSP_Length(dimension))
        }
        var exponent = 1.0
        vDSP_dotprD(partial, 1, relative, 1, &exponent, vDSP_Length(dimension))
        exponent *= -0.5
        
        return exp(exponent) * multiplier
    }
    
    ///  Function to get a set of random vectors
    ///  Setup is computationaly expensive, so call once to get multiple vectors
    open func random(_ count: Int) throws -> [[Double]] {
        var sqrtEigenValues = [Double](repeating: 0.0, count: dimension)
        var translationMatrix = [Double](repeating: 0.0, count: dimension*dimension)
        if (diagonalΣ) {
            //  eigenValues are the diagonals - get sqrt of them for multiplication
            for element in 0..<dimension {
                sqrtEigenValues[element] = sqrt(Σ[element])
            }
        }
        else {
            //  If a non-diagonal covariance matrix, get the eigenvalues and eigenvectors
            //  Get the SVD decomposition of the Σ matrix
            let jobZChar = "S" as NSString
            var jobZ : Int8 = Int8(jobZChar.character(at: 0))          //  return min(m,n) rows of Σ
            var n : __CLPK_integer = __CLPK_integer(dimension)
            var u = [Double](repeating: 0.0, count: dimension * dimension)
            var work : [Double] = [0.0]
            var lwork : __CLPK_integer = -1        //  Ask for the best size of the work array
            let iworkSize = 8 * dimension
            var iwork = [__CLPK_integer](repeating: 0, count: iworkSize)
            var info : __CLPK_integer = 0
            var A = Σ       //  Leave Σ intact
            var eigenValues = [Double](repeating: 0.0, count: dimension)
            var eigenVectors = [Double](repeating: 0.0, count: dimension*dimension)
            dgesdd_(&jobZ, &n, &n, &A, &n, &eigenValues, &u, &n, &eigenVectors, &n, &work, &lwork, &iwork, &info)
            if (info != 0 || work[0] < 1) {
                throw GaussianError.errorInSVDParameters
            }
            lwork = __CLPK_integer(work[0])
            work = [Double](repeating: 0.0, count: Int(work[0]))
            dgesdd_(&jobZ, &n, &n, &A, &n, &eigenValues, &u, &n, &eigenVectors, &n, &work, &lwork, &iwork, &info)
            if (info < 0) {
                throw GaussianError.errorInSVDParameters
            }
            if (info > 0) {
                throw GaussianError.svdDidNotConverge
            }
            
            //  Extract the eigenvectors multiplied by the square root of the eigenvalues - make a row-major matrix for dataset vector multiplication using vDSP
            for vector in 0..<dimension {
                let sqrtEigenValue = sqrt(eigenValues[vector])
                for column in 0..<dimension {
                    translationMatrix[(vector * dimension) + column] = eigenValues[vector + (column * dimension)] * sqrtEigenValue
                }
            }
        }
        
        //  Get a set of vectors
        var results : [[Double]] = []
        for _ in 0..<count {
            //  Get random uniform vector
            var entry = [Double](repeating: 0.0, count: dimension)
            for element in 0..<dimension {
                entry[element] = Gaussian.gaussianRandom(0.0, standardDeviation: 1.0)
            }
            
            //  Extend vector based on the covariance matrix
            if (diagonalΣ) {
                //  Since diagonal, the eigenvectors are unit vectors, so just multiply each element by the square root of the eigenvalues - which are the diagonal elements
                vDSP_vmulD(entry, 1, sqrtEigenValues, 1, &entry, 1, vDSP_Length(dimension))
            }
            else {
                vDSP_mmulD(translationMatrix, 1, entry, 1, &entry, 1, vDSP_Length(dimension), vDSP_Length(1), vDSP_Length(dimension))
            }
            
            //  Add the mean
            vDSP_vaddD(entry, 1, μ, 1, &entry, 1, vDSP_Length(dimension))
            
            //  Insert vector into return results
            results.append((entry))
        }
        return results
    }
}
#endif
