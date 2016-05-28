//
//  MixtureOfGaussians.swift
//  AIToolbox
//
//  Created by Kevin Coble on 4/12/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation
import Accelerate


public enum MixtureOfGaussianError: ErrorType {
    case KMeansFailed
}


///  Class for mixture-of-gaussians density function learned from data
///     model is α₁𝒩(μ₁, Σ₁) + α₂𝒩(μ₂, Σ₂) + α₃𝒩(μ₃, Σ₃) + ...
///     Output is a single Double value that is the probability for for the input vector
public class MixtureOfGaussians : Regressor
{
    public private(set) var inputDimension : Int
    public private(set) var termCount : Int
    public private(set) var diagonalΣ : Bool
    public private(set) var gaussians : [Gaussian] = []
    public private(set) var mvgaussians : [MultivariateGaussian] = []
    public var α : [Double]
    public var initWithKMeans = true        //  if true initial probability distributions are computed with kmeans of training data, else random
    public var convergenceLimit = 0.0000001
    var initializeFunction : ((trainData: DataSet)->[Double])!
    
    public init(inputSize: Int, numberOfTerms: Int, diagonalCoVariance: Bool) throws
    {
        inputDimension = inputSize
        termCount = numberOfTerms
        diagonalΣ = diagonalCoVariance
        
        do {
            for _ in 0..<termCount {
                if (inputDimension == 1) {
                    try gaussians.append(Gaussian(mean: 0.0, variance: 1.0))
                }
                else {
                    try mvgaussians.append(MultivariateGaussian(dimension: inputDimension, diagonalCovariance: diagonalΣ))
                }
            }
        }
        catch let error {
            throw error
        }
        
        α = [Double](count: termCount, repeatedValue: 1.0)
    }
    
    public func getInputDimension() -> Int
    {
        return inputDimension
    }
    public func getOutputDimension() -> Int
    {
        return 1
    }
    public func setParameters(parameters: [Double]) throws
    {
        if (parameters.count < getParameterDimension()) { throw MachineLearningError.NotEnoughData }
        
        var offset = 0
        for index in 0..<termCount {
            α[index] = parameters[offset]
            offset += 1
            if (inputDimension == 1) {
                try gaussians[index].setParameters(Array(parameters[offset..<offset+2]))
                offset += 2
            }
            else {
                let neededSize = mvgaussians[index].getParameterDimension()
                try mvgaussians[index].setParameters(Array(parameters[offset..<offset+neededSize]))
                offset += neededSize
            }
        }
    }
    public func getParameterDimension() -> Int
    {
        var parameterCount = 1 + inputDimension     //  alpha and mean
        if (diagonalΣ) {
            parameterCount += inputDimension    //  If diagonal, only one covariance term per input
        }
        else {
            parameterCount += inputDimension * inputDimension    //  If not diagonal, full matrix of covariance
        }
        
        parameterCount *= termCount     //  alpha, mean, and covariance for each gaussian term
        
        return parameterCount
    }
    
    ///  A custom initializer for Mixture-of-Gaussians must assign the training data to classes [the 'classes' member of the dataset]
    public func setCustomInitializer(function: ((trainData: DataSet)->[Double])!) {
        initializeFunction = function
    }
    
    public func getParameters() throws -> [Double]
    {
        var parameters : [Double] = []
        for index in 0..<termCount {
            if (inputDimension == 1) {
                parameters += try gaussians[index].getParameters()
            }
            else {
                parameters += try mvgaussians[index].getParameters()
            }
        }
        return parameters
    }
    
    ///  Function to calculate the parameters of the model
    public func trainRegressor(trainData: DataSet) throws
    {
        //  Verify that the data is regression data
        if (trainData.dataType != DataSetType.Regression) { throw MachineLearningError.DataNotRegression }
        if (trainData.inputDimension != inputDimension) { throw MachineLearningError.DataWrongDimension }
        if (trainData.outputDimension != 1) { throw MachineLearningError.DataWrongDimension }
        if (trainData.size < termCount) { throw MachineLearningError.NotEnoughData }
        
        //  Initialize the gaussians and weights
        //  Algorithm from http://arxiv.org/pdf/1312.5946.pdf
        var centroids : [[Double]] = []
        for _ in 0..<termCount {
            centroids.append([Double](count: inputDimension, repeatedValue: 0.0))
        }
        
        if (initWithKMeans) {
            //  Group the points using KMeans
            let kmeans = KMeans(classes: termCount)
            do {
                try kmeans.train(trainData)
            }
            catch {
                throw MixtureOfGaussianError.KMeansFailed
            }
            centroids = kmeans.centroids
        }
        else {
            //  Assign each point to a random term
            if let initFunc = initializeFunction {
                //  If a custom initializer has been provided, use it to assign the initial classes
                initFunc(trainData: trainData)
            }
            else {
                //  No initializer, assign to random classes
                var classes = [Int](count: trainData.size, repeatedValue: 0)
                for index in 0..<termCount {        //  Assign first few points to each class to guarantee at least one point per term
                    classes[index] = index
                }
                for index in termCount..<trainData.size {
                    classes[index] = Int(arc4random_uniform(UInt32(termCount)))
                }
                trainData.classes = classes
            }
            
            //  Calculate centroids
            var counts = [Int](count: termCount, repeatedValue: 0)
            for point in 0..<trainData.size {
                let pointClass = trainData.classes![point]
                counts[pointClass] += 1
                let ptr = UnsafeMutablePointer<Double>(centroids[pointClass])      //  Swift bug!  can't put this in line!
                vDSP_vaddD(trainData.inputs[point], 1, centroids[pointClass], 1, ptr, 1, vDSP_Length(inputDimension))
            }
            for term in 0..<termCount {
                var inverse = 1.0 / Double(counts[term])
                let ptr = UnsafeMutablePointer<Double>(centroids[term])      //  Swift bug!  can't put this in line!
                vDSP_vsmulD(centroids[term], 1, &inverse, ptr, 1, vDSP_Length(inputDimension * inputDimension))
            }
        }
        
        //  Get the counts and sum the covariance terms
        var counts = [Int](count: termCount, repeatedValue: 0)
        var covariance : [[Double]] = []
        var sphericalCovariance = [Double](count: inputDimension, repeatedValue: 0.0)
        for _ in 0..<termCount { covariance.append([Double](count: inputDimension * inputDimension, repeatedValue: 0.0))}
        var offset = [Double](count: inputDimension, repeatedValue: 0.0)
        var matrix = [Double](count: inputDimension * inputDimension, repeatedValue: 0.0)
        for point in 0..<trainData.size {
            //  Count
            let pointClass = trainData.classes![point]
            counts[pointClass] += 1
            //  Get the distance from the mean
            vDSP_vsubD(centroids[pointClass], 1, trainData.inputs[point], 1, &offset, 1, vDSP_Length(inputDimension))
            if (!diagonalΣ) {   //  We will force into spherical if diagonal covariance
                //  Multiply into covariance matrix and sum
                vDSP_mmulD(offset, 1, offset, 1, &matrix, 1, vDSP_Length(inputDimension), vDSP_Length(inputDimension), vDSP_Length(1))
                let ptr = UnsafeMutablePointer<Double>(covariance[pointClass])      //  Swift bug!  can't put this in line!
                vDSP_vaddD(matrix, 1, covariance[pointClass], 1, ptr, 1, vDSP_Length(inputDimension * inputDimension))
            }
            //  Get dot product and sum into spherical in case covariance matrix is not positive definite  (dot product of offset is distance squared
            var dotProduct = 0.0
            vDSP_dotprD(offset, 1, offset, 1, &dotProduct, vDSP_Length(inputDimension))
            sphericalCovariance[pointClass] += dotProduct
        }
        
        //  If multivariate, verify positive-definite covariance matrix, or do same processing for diagonal covariance
        if (inputDimension > 1 || diagonalΣ) {
            var nonSPD = false
            if (!diagonalΣ) {
                //  If not diagonal, check for positive definite
                for term in 0..<termCount {
                    let uploChar = "U" as NSString
                    var uplo : Int8 = Int8(uploChar.characterAtIndex(0))          //  use upper triangle
                    var A = covariance[term]       //  Make a copy so it isn't mangled
                    var n : Int32 = Int32(inputDimension)
                    var info : Int32 = 0
                    dpotrf_(&uplo, &n, &A, &n, &info)
                    if (info != 0) {
                        nonSPD = true
                        break
                    }
                }
            }
            
            //  If not positive-definite (or diagonal covariance), use spherical covariance
            if (nonSPD || diagonalΣ) {
                //  Set each covariance matrix to the identity matrix times the sum of dotproduct of distances
                for term in 0..<termCount {
                    covariance[term] = [Double](count: inputDimension * inputDimension, repeatedValue: 0.0)
                    let scale = 1.0 / Double(inputDimension)
                    for row in 0..<inputDimension {
                        covariance[term][row * inputDimension + row] = sphericalCovariance[term] * scale
                    }
                }
            }
            
            //  The stated algorithm continues with another positive-definite check, but a positive constant times the identity matrix should always be positive definite, so I am stopping here
        }
        
        //  Assign the value to the gaussians
        for term in 0..<termCount {
            α[term] = Double(counts[term]) / Double(trainData.size)
            if (counts[term] > 1) {
                var inverse = 1.0 / Double(counts[term])
                let ptr = UnsafeMutablePointer<Double>(covariance[term])      //  Swift bug!  can't put this in line!
                vDSP_vsmulD(covariance[term], 1, &inverse, ptr, 1, vDSP_Length(inputDimension * inputDimension))
            }
            if (inputDimension == 1) {
                gaussians[term].setMean(centroids[term][0])
                gaussians[term].setVariance(covariance[term][0])
            }
            else {
                do {
                    try mvgaussians[term].setMean(centroids[term])
                    if (diagonalΣ) {
                        var diagonalTerms : [Double] = []
                        for row in 0..<inputDimension {
                            diagonalTerms.append(covariance[term][row * inputDimension + row])
                        }
                        try mvgaussians[term].setCovarianceMatrix(diagonalTerms)
                    }
                    else {
                        try mvgaussians[term].setCovarianceMatrix(covariance[term])
                    }
                }
                catch let error {
                    throw error
                }
            }
        }
        
        //  Use the continue function to converge the model
        do {
            try continueTrainingRegressor(trainData)
        }
        catch let error {
            throw error
        }
    }
    
    ///  Function to continue calculating the parameters of the model with more data, without initializing parameters
    public func continueTrainingRegressor(trainData: DataSet) throws
    {
        //  Verify that the data is regression data
        if (trainData.dataType != DataSetType.Regression) { throw MachineLearningError.DataNotRegression }
        if (trainData.inputDimension != inputDimension) { throw MachineLearningError.DataWrongDimension }
        if (trainData.outputDimension != 1) { throw MachineLearningError.DataWrongDimension }
        
        //  Calculate the log-likelihood with the current parameters for the convergence check
        var lastLogLikelihood = 0.0
        for point in 0..<trainData.size {
            do {
                let sum = try predictOne(trainData.inputs[point])
                lastLogLikelihood += log(sum[0])
            }
            catch let error {
                throw error
            }
        }
        
        //  Use the EM algorithm until it converges
        var membershipWeights : [[Double]]
        
        var centroids : [[Double]] = []
        var matrix = [Double](count: inputDimension * inputDimension, repeatedValue: 0.0)
        var difference = convergenceLimit + 1.0
        while (difference > convergenceLimit) {    //  Go till convergence
            difference = 0
            
            // 'E' step
            //  Clear the membership weights
            membershipWeights = []
            for _ in 0..<termCount {
                membershipWeights.append([Double](count: trainData.size, repeatedValue: 0.0))
            }
            
            //  Calculate the membership weights
            var total : Double
            var probability : Double
            for point in 0..<trainData.size {
                total = 0.0
                for term in 0..<termCount {
                    if (inputDimension == 1) {
                        probability = α[term] * gaussians[term].getProbability(trainData.inputs[point][0])
                    }
                    else {
                        probability = try α[term] * mvgaussians[term].getProbability(trainData.inputs[point])
                    }
                    total += probability
                    membershipWeights[term][point] = probability
                }
                let scale = 1.0 / total
                for term in 0..<termCount {
                    membershipWeights[term][point] *= scale
                }
            }
            
            // 'M' step
            //  Calculate α's and μ's
            var weightTotals = [Double](count: termCount, repeatedValue: 0.0)
            var vector = [Double](count: inputDimension, repeatedValue: 0.0)
            centroids = []
            for term in 0..<termCount {
                centroids.append([Double](count: inputDimension, repeatedValue: 0.0))
                for point in 0..<trainData.size {
                    weightTotals[term] += membershipWeights[term][point]
                    vDSP_vsmulD(trainData.inputs[point], 1, &membershipWeights[term][point], &vector, 1, vDSP_Length(inputDimension))
                    let ptr = UnsafeMutablePointer<Double>(centroids[term])      //  Swift bug!  can't put this in line!
                    vDSP_vaddD(vector, 1, centroids[term], 1, ptr, 1, vDSP_Length(inputDimension))
                }
                α[term] = weightTotals[term] / Double(trainData.size)
                var scale = 1.0 / weightTotals[term]
                let ptr = UnsafeMutablePointer<Double>(centroids[term])      //  Swift bug!  can't put this in line!
                vDSP_vsmulD(centroids[term], 1, &scale, ptr, 1, vDSP_Length(inputDimension))
                if (inputDimension == 1) {
                    gaussians[term].setMean(centroids[term][0])
                }
                else {
                    do {
                        try mvgaussians[term].setMean(centroids[term])
                    }
                    catch let error {
                        throw error
                    }
                }
            }
            
            //  Calculate the covariance matrices
            if (diagonalΣ) {
                for term in 0..<termCount {
                    var sphericalCovarianceValue = 0.0
                    for point in 0..<trainData.size {
                        //  Get difference vector
                        vDSP_vsubD(trainData.inputs[point], 1, centroids[term], 1, &vector, 1, vDSP_Length(inputDimension))
                        var dotProduct = 0.0
                        vDSP_dotprD(vector, 1, vector, 1, &dotProduct, vDSP_Length(inputDimension))
                        sphericalCovarianceValue += membershipWeights[term][point] * dotProduct
                    }
                    sphericalCovarianceValue /= (weightTotals[term] * Double(inputDimension))
                    var Σ = [Double](count: inputDimension, repeatedValue: sphericalCovarianceValue)
                    if (inputDimension == 1) {
                        gaussians[term].setVariance(Σ[0])
                    }
                    else {
                        do {
                            try mvgaussians[term].setCovarianceMatrix(Σ)
                        }
                        catch let error {
                            throw error
                        }
                    }
                }
            }
            else {
                for term in 0..<termCount {
                    var Σ = [Double](count: inputDimension * inputDimension, repeatedValue: 0.0)
                    for point in 0..<trainData.size {
                        //  Get difference vector
                        vDSP_vsubD(trainData.inputs[point], 1, centroids[term], 1, &vector, 1, vDSP_Length(inputDimension))
                        //  Multiply by transpose to get covariance factor
                        vDSP_mmulD(vector, 1, vector, 1, &matrix, 1, vDSP_Length(inputDimension), vDSP_Length(inputDimension), vDSP_Length(1))
                        //  Weight by the point's membership
                        vDSP_vsmulD(matrix, 1, &membershipWeights[term][point], &matrix, 1, vDSP_Length(inputDimension * inputDimension))
                        //  Accumulate
                        vDSP_vaddD(matrix, 1, Σ, 1, &Σ, 1, vDSP_Length(inputDimension * inputDimension))
                        //  Divide by total weight to get average covariance
                        var scale = 1.0 / weightTotals[term]
                        vDSP_vsmulD(Σ, 1, &scale, &Σ, 1, vDSP_Length(inputDimension * inputDimension))
                    }
                    if (inputDimension == 1) {
                        gaussians[term].setVariance(Σ[0])
                    }
                    else {
                        do {
                            try mvgaussians[term].setCovarianceMatrix(Σ)
                        }
                        catch let error {
                            throw error
                        }
                    }
                }
            }
            
            //  Calculate the log-likelihood to see if we have converged
            var logLikelihood = 0.0
            for point in 0..<trainData.size {
                do {
                    let sum = try predictOne(trainData.inputs[point])
                    logLikelihood += log(sum[0])
                }
                catch let error {
                    throw error
                }
            }
            
            difference = fabs(lastLogLikelihood - logLikelihood)
            lastLogLikelihood = logLikelihood
        }
    }
    
    ///  Function to calculate the result from an input vector   
    public func predictOne(inputs: [Double]) throws ->[Double]
    {
        if (inputs.count != inputDimension) { throw MachineLearningError.DataWrongDimension }
        
        var result = 0.0
        do {
            for index in 0..<termCount {
                if (inputDimension == 1) {
                    result += α[index] * gaussians[index].getProbability(inputs[0])
                }
                else {
                    result += try α[index] * mvgaussians[index].getProbability(inputs)
                }
            }
        }
        catch let error {
            throw error
        }
        
        return [result]
    }
    
    public func predict(testData: DataSet) throws
    {
        //  Verify the data set is the right type
        if (testData.dataType != .Regression) { throw MachineLearningError.DataNotRegression }
        if (testData.inputDimension != inputDimension) { throw MachineLearningError.DataWrongDimension }
        
        //  predict on each input
        testData.outputs = []
        for index in 0..<testData.size {
            do {
                try testData.outputs!.append(predictOne(testData.inputs[index]))
            }
            catch let error {
                throw error
            }
        }
        
    }
}