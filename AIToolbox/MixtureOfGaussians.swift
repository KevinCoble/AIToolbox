//
//  MixtureOfGaussians.swift
//  AIToolbox
//
//  Created by Kevin Coble on 4/12/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation
import Accelerate


public enum MixtureOfGaussianError: Error {
    case kMeansFailed
}


///  Class for mixture-of-gaussians density function learned from data
///     model is α₁𝒩(μ₁, Σ₁) + α₂𝒩(μ₂, Σ₂) + α₃𝒩(μ₃, Σ₃) + ...
///     Output is a single Double value that is the probability for for the input vector
open class MixtureOfGaussians : Regressor
{
    open fileprivate(set) var inputDimension : Int
    open fileprivate(set) var termCount : Int
    open fileprivate(set) var diagonalΣ : Bool
    open fileprivate(set) var gaussians : [Gaussian] = []
    open fileprivate(set) var mvgaussians : [MultivariateGaussian] = []
    open var α : [Double]
    open var initWithKMeans = true        //  if true initial probability distributions are computed with kmeans of training data, else random
    open var convergenceLimit = 0.0000001
    var initializeFunction : ((_ trainData: MLDataSet)->[Double])!
    
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
        
        α = [Double](repeating: 1.0, count: termCount)
    }
    
    open func getInputDimension() -> Int
    {
        return inputDimension
    }
    open func getOutputDimension() -> Int
    {
        return 1
    }
    open func setParameters(_ parameters: [Double]) throws
    {
        if (parameters.count < getParameterDimension()) { throw MachineLearningError.notEnoughData }
        
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
    open func getParameterDimension() -> Int
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
    open func setCustomInitializer(_ function: ((_ trainData: MLDataSet)->[Double])!) {
        initializeFunction = function
    }
    
    open func getParameters() throws -> [Double]
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
    open func trainRegressor(_ trainData: MLRegressionDataSet) throws
    {
        //  Verify that the data is regression data
        if (trainData.dataType != .regression) { throw MachineLearningError.dataNotRegression }
        if (trainData.inputDimension != inputDimension) { throw MachineLearningError.dataWrongDimension }
        if (trainData.outputDimension != 1) { throw MachineLearningError.dataWrongDimension }
        if (trainData.size < termCount) { throw MachineLearningError.notEnoughData }
        
        //  Initialize the gaussians and weights
        //  Algorithm from http://arxiv.org/pdf/1312.5946.pdf
        var centroids : [[Double]] = []
        for _ in 0..<termCount {
            centroids.append([Double](repeating: 0.0, count: inputDimension))
        }
        
        //  Make a classification data set from the regression set
        if let classificationDataSet = DataSet(dataType : .classification, withInputsFrom: trainData) {
            
            if (initWithKMeans) {
                //  Group the points using KMeans
                let kmeans = KMeans(classes: termCount)
                do {
                    try kmeans.train(classificationDataSet)
                }
                catch {
                    throw MixtureOfGaussianError.kMeansFailed
                }
                centroids = kmeans.centroids
            }
            else {
                //  Assign each point to a random term
                if let initFunc = initializeFunction {
                    //  If a custom initializer has been provided, use it to assign the initial classes
                    _ = initFunc(trainData)
                }
                else {
                    //  No initializer, assign to random classes
                    for index in 0..<termCount {        //  Assign first few points to each class to guarantee at least one point per term
                        try classificationDataSet.setClass(index, newClass: index)
                    }
                    for index in termCount..<trainData.size {
                        try classificationDataSet.setClass(index, newClass: Int(arc4random_uniform(UInt32(termCount))))
                    }
                }
                
                //  Calculate centroids
                var counts = [Int](repeating: 0, count: termCount)
                for point in 0..<trainData.size {
                    let pointClass = try classificationDataSet.getClass(point)
                    let inputs = try trainData.getInput(point)
                    counts[pointClass] += 1
                    
                    withUnsafePointer(to: &centroids[pointClass][0]) {
                        vDSP_vaddD(inputs, 1, $0, 1, UnsafeMutablePointer<Double>(mutating: $0) , 1, vDSP_Length(inputDimension))
                    }
                    
                }
                for term in 0..<termCount {
                    var inverse = 1.0 / Double(counts[term])
                    withUnsafePointer(to: &centroids[term][0]) {
                        vDSP_vsmulD($0, 1, &inverse, UnsafeMutablePointer<Double>(mutating: $0) , 1, vDSP_Length(inputDimension * inputDimension))
                    }
                }
            }
            
            //  Get the counts and sum the covariance terms
            var counts = [Int](repeating: 0, count: termCount)
            var covariance : [[Double]] = []
            var sphericalCovariance = [Double](repeating: 0.0, count: inputDimension)
            for _ in 0..<termCount { covariance.append([Double](repeating: 0.0, count: inputDimension * inputDimension))}
            var offset = [Double](repeating: 0.0, count: inputDimension)
            var matrix = [Double](repeating: 0.0, count: inputDimension * inputDimension)
            for point in 0..<trainData.size {
                //  Count
                let pointClass = try classificationDataSet.getClass(point)
                let inputs = try trainData.getInput(point)
                counts[pointClass] += 1
                //  Get the distance from the mean
                vDSP_vsubD(centroids[pointClass], 1, inputs, 1, &offset, 1, vDSP_Length(inputDimension))
                if (!diagonalΣ) {   //  We will force into spherical if diagonal covariance
                    //  Multiply into covariance matrix and sum
                    vDSP_mmulD(offset, 1, offset, 1, &matrix, 1, vDSP_Length(inputDimension), vDSP_Length(inputDimension), vDSP_Length(1))
                    withUnsafePointer(to: &covariance[pointClass][0]) {
                        vDSP_vaddD(matrix, 1, $0, 1, UnsafeMutablePointer<Double>(mutating: $0), 1, vDSP_Length(inputDimension * inputDimension))
                    }
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
                        var uplo : Int8 = Int8(uploChar.character(at: 0))          //  use upper triangle
                        var A = covariance[term]       //  Make a copy so it isn't mangled
                        var n : __CLPK_integer = __CLPK_integer(inputDimension)
                        var lda = n
                        var info : __CLPK_integer = 0
                        dpotrf_(&uplo, &n, &A, &lda, &info)
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
                        covariance[term] = [Double](repeating: 0.0, count: inputDimension * inputDimension)
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
                    withUnsafePointer(to: &covariance[term][0]) {
                        vDSP_vsmulD($0, 1, &inverse, UnsafeMutablePointer<Double>(mutating: $0), 1, vDSP_Length(inputDimension * inputDimension))
                    }
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
    open func continueTrainingRegressor(_ trainData: MLRegressionDataSet) throws
    {
        //  Verify that the data is regression data
        if (trainData.dataType != DataSetType.regression) { throw MachineLearningError.dataNotRegression }
        if (trainData.inputDimension != inputDimension) { throw MachineLearningError.dataWrongDimension }
        if (trainData.outputDimension != 1) { throw MachineLearningError.dataWrongDimension }
        
        //  Calculate the log-likelihood with the current parameters for the convergence check
        var lastLogLikelihood = 0.0
        for point in 0..<trainData.size {
            do {
                let inputs = try trainData.getInput(point)
                let sum = try predictOne(inputs)
                lastLogLikelihood += log(sum[0])
            }
            catch let error {
                throw error
            }
        }
        
        //  Use the EM algorithm until it converges
        var membershipWeights : [[Double]]
        
        var centroids : [[Double]] = []
        var matrix = [Double](repeating: 0.0, count: inputDimension * inputDimension)
        var difference = convergenceLimit + 1.0
        while (difference > convergenceLimit) {    //  Go till convergence
            difference = 0
            
            // 'E' step
            //  Clear the membership weights
            membershipWeights = []
            for _ in 0..<termCount {
                membershipWeights.append([Double](repeating: 0.0, count: trainData.size))
            }
            
            //  Calculate the membership weights
            var total : Double
            var probability : Double
            for point in 0..<trainData.size {
                total = 0.0
                for term in 0..<termCount {
                    let inputs = try trainData.getInput(point)
                    if (inputDimension == 1) {
                        probability = α[term] * gaussians[term].getProbability(inputs[0])
                    }
                    else {
                        probability = try α[term] * mvgaussians[term].getProbability(inputs)
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
            var weightTotals = [Double](repeating: 0.0, count: termCount)
            var vector = [Double](repeating: 0.0, count: inputDimension)
            centroids = []
            for term in 0..<termCount {
                centroids.append([Double](repeating: 0.0, count: inputDimension))
                for point in 0..<trainData.size {
                    let inputs = try trainData.getInput(point)
                    weightTotals[term] += membershipWeights[term][point]
                    vDSP_vsmulD(inputs, 1, &membershipWeights[term][point], &vector, 1, vDSP_Length(inputDimension))
                    withUnsafePointer(to: &centroids[term][0]) {
                        vDSP_vaddD(vector, 1, $0, 1, UnsafeMutablePointer<Double>(mutating: $0), 1, vDSP_Length(inputDimension))
                    }
                }
                α[term] = weightTotals[term] / Double(trainData.size)
                var scale = 1.0 / weightTotals[term]
                withUnsafePointer(to: &centroids[term][0]) {
                    vDSP_vsmulD($0, 1, &scale, UnsafeMutablePointer<Double>(mutating: $0), 1, vDSP_Length(inputDimension))
                }
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
                        let inputs = try trainData.getInput(point)
                        vDSP_vsubD(inputs, 1, centroids[term], 1, &vector, 1, vDSP_Length(inputDimension))
                        var dotProduct = 0.0
                        vDSP_dotprD(vector, 1, vector, 1, &dotProduct, vDSP_Length(inputDimension))
                        sphericalCovarianceValue += membershipWeights[term][point] * dotProduct
                    }
                    sphericalCovarianceValue /= (weightTotals[term] * Double(inputDimension))
                    let Σ = [Double](repeating: sphericalCovarianceValue, count: inputDimension)
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
                    var Σ = [Double](repeating: 0.0, count: inputDimension * inputDimension)
                    for point in 0..<trainData.size {
                        //  Get difference vector
                        let inputs = try trainData.getInput(point)
                        vDSP_vsubD(inputs, 1, centroids[term], 1, &vector, 1, vDSP_Length(inputDimension))
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
                    let inputs = try trainData.getInput(point)
                    let sum = try predictOne(inputs)
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
    open func predictOne(_ inputs: [Double]) throws ->[Double]
    {
        if (inputs.count != inputDimension) { throw MachineLearningError.dataWrongDimension }
        
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
    
    open func predict(_ testData: MLRegressionDataSet) throws
    {
        //  Verify the data set is the right type
        if (testData.dataType != .regression) { throw MachineLearningError.dataNotRegression }
        if (testData.inputDimension != inputDimension) { throw MachineLearningError.dataWrongDimension }
        
        //  predict on each input
        for index in 0..<testData.size {
            do {
                let inputs = try testData.getInput(index)
                try testData.setOutput(index, newOutput: predictOne(inputs))
            }
            catch let error {
                throw error
            }
        }
        
    }
}
