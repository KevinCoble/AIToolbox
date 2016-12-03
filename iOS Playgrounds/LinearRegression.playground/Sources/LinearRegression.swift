//
//  File.swift
//  AIToolbox
//
//  Created by Kevin Coble on 4/4/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation
import Accelerate


public enum LinearRegressionError: Error {
    case modelExceedsInputDimension
    case matrixSolutionError
    case negativeInLogOrPower
    case divideByZero
}


///  Function for subterm
public enum SubtermFunction { case
    none,
    naturalExponent,
    sine,
    cosine,
    log,
    power       //  uses power value of term raised to input
}

///  Struct for a piece of a term of the linear equation being fitted
///  Subterms can be combined -->  a₁²a₂² is two subterms, a₁² mulitplied by a₂² in terms
public struct LinearRegressionSubTerm
{
    let inputIndex : Int
    public var power = 1.0
    public var function = SubtermFunction.none
    public var divide = false      //  If true this term divides the previous subterm (ignored on first subterm), else multiply
    
    public init(withInput: Int)
    {
        inputIndex = withInput
    }
    
    public func getSubTermValue(_ withInputs: [Double]) throws -> Double
    {
        var result = withInputs[inputIndex]
        if (power != 1.0) {
            let powerResult = pow(result, power)        //  Allow negatives if power is integer - if not, nan will result
            if (result < 0.0 && powerResult.isNaN) { throw LinearRegressionError.negativeInLogOrPower }
            result = powerResult
        }
        switch (function) {
        case .none:
            break
        case .naturalExponent:
            result = exp(result)
        case .sine:
            result = sin(result)
        case .cosine:
            result = cos(result)
        case .log:
            if (result < 0.0) { throw LinearRegressionError.negativeInLogOrPower }
            result = log(result)
        case .power:       //  uses power value of term raised to input
            if (power < 0.0) { throw LinearRegressionError.negativeInLogOrPower }
            result = pow(power, withInputs[inputIndex])
        }
        return result
    }
}

///  Struct for a single term of the linear equation being fitted
///     each term is a set of LinearRegressionSubTerms multiplied (or divided) together
public struct LinearRegressionTerm
{
    var subterms : [LinearRegressionSubTerm] = []
    
    public init()
    {
        //  Empty initializer - all terms must be added manually
    }
    
    public init(withInput: Int, atPower: Double)
    {
        var subTerm = LinearRegressionSubTerm(withInput: withInput)
        subTerm.power = atPower
        subterms.append(subTerm)
    }
    
    public mutating func addSubTerm(_ subTerm: LinearRegressionSubTerm)
    {
        subterms.append(subTerm)
    }
    
    func getHighestInputIndex() -> Int
    {
        var highestIndex = -9999
        for subterm in subterms {
            if (subterm.inputIndex > highestIndex) { highestIndex = subterm.inputIndex }
        }
        
        return highestIndex
    }
    
    public func getTermValue(_ withInputs: [Double]) throws -> Double
    {
        var result = 1.0
        
        for subterm in subterms {
            if (subterm.divide) {
                do {
                    let value = try subterm.getSubTermValue(withInputs)
                    if (value != 0) {
                        result /= value
                    }
                    else {
                        throw LinearRegressionError.divideByZero
                    }
                }
                catch let error {
                    throw error
                }
            }
            else {
                do {
                    try result *= subterm.getSubTermValue(withInputs)
                }
                catch let error {
                    throw error
                }
            }
        }
        
        return result
    }
}

///  Class for linear (in parameter space) expression
///     model is (optional bias) + At₁ + Bt₂ + Ct₃ + ...  where t are the LinearRegressionTerms
open class LinearRegressionModel : Regressor
{
    open fileprivate(set) var inputDimension : Int
    open fileprivate(set) var outputDimension : Int
    var terms : [LinearRegressionTerm] = []
    open var includeBias = true
    open var regularization : Double?
    open var Θ : [[Double]] = []
    
    public init(inputSize: Int, outputSize: Int)
    {
        inputDimension = inputSize
        outputDimension = outputSize
    }
    
    public convenience init(inputSize: Int, outputSize: Int, polygonOrder: Int, includeCrossTerms: Bool = false)
    {
        self.init(inputSize: inputSize, outputSize: outputSize)
        
        //  Add terms for each input to the polygon order specified
        for input in 0..<inputSize {
            for power in 0..<polygonOrder {
                let term = LinearRegressionTerm(withInput: input, atPower: Double(power+1))
                terms.append(term)
                if (includeCrossTerms) {
                    for otherInput in 0..<input {
                        for otherPower in 0..<polygonOrder {
                            var term = LinearRegressionTerm(withInput: input, atPower: Double(power+1))
                            var subTerm = LinearRegressionSubTerm(withInput: otherInput)
                            subTerm.power = Double(otherPower+1)
                            term.addSubTerm(subTerm)
                            terms.append(term)
                        }
                    }
               }
            }
        }
    }
    
    open func addTerm(_ newTerm: LinearRegressionTerm)
    {
        terms.append(newTerm)
    }
    
    open func getInputDimension() -> Int
    {
        return inputDimension
    }
    
    open func getOutputDimension() -> Int
    {
        return outputDimension
    }
    
    open func getParameterDimension() -> Int
    {
        var numParameters = terms.count
        if (includeBias) { numParameters += 1 }
        numParameters *= outputDimension
        return numParameters
    }
    
    open func setParameters(_ parameters: [Double]) throws
    {
        if (parameters.count < getParameterDimension()) { throw MachineLearningError.notEnoughData }
        
        var numParameters = terms.count
        if (includeBias) { numParameters += 1 }
        var offset = 0
        if (Θ.count < outputDimension) { Θ = [[Double]](repeating: [], count: outputDimension) }
        for index in 0..<outputDimension {
            Θ[index] = Array(parameters[offset..<(offset+numParameters)])
            offset += numParameters
        }
    }
    
    open func setCustomInitializer(_ function: ((_ trainData: MLDataSet)->[Double])!) {
        //  Ignore, as Linear Regression doesn't use an initialization
    }
    
    open func getParameters() throws -> [Double]
    {
        var parameters : [Double] = []
        for index in 0..<outputDimension {
            parameters += Θ[index]
        }
        return parameters
    }

    open func trainRegressor(_ trainData: MLRegressionDataSet) throws
    {
        //  Verify that the data is regression data
        if (trainData.dataType != DataSetType.regression) { throw MachineLearningError.dataNotRegression }
        
        //  Get the number of input values needed by the model
        var neededInputs = 0
        for term in terms {
            let termHighest = term.getHighestInputIndex()
            if (termHighest > neededInputs) { neededInputs = termHighest }
        }
        
        //  Validate that the model has the dimension
        if (neededInputs >= inputDimension) {
            throw LinearRegressionError.modelExceedsInputDimension
        }
        
        //  Validate the data size
        if (trainData.inputDimension != inputDimension || trainData.outputDimension != outputDimension) {
            throw MachineLearningError.dataWrongDimension
        }
        
        //  Get the number of terms in the matrix (columns)
        let numColumns = getParameterDimension()
        
        //  Make sure we have enough data for a solution (at least 1 per term)
        if (trainData.size < numColumns) {
            throw MachineLearningError.notEnoughData
        }
        
        //  Allocate the parameter array
        Θ = [[Double]](repeating: [], count: outputDimension)
        
        //  We can use lapack dgels if no regularization term
        if (regularization == nil) {
       
            //  Make a column-major matrix of the data, with any bias term
            var A = [Double](repeating: 0.0, count: trainData.size * numColumns)
            var offset = 0
            if (includeBias) {
                for _ in 0..<trainData.size {
                    A[offset] = 1.0
                    offset += 1
                }
            }
            for parameter in 0..<terms.count {
                for index in 0..<trainData.size {
                    do {
                        let inputs = try trainData.getInput(index)
                        A[offset] = try terms[parameter].getTermValue(inputs)
                    }
                    catch let error {
                        throw error
                    }
                    offset += 1
                }
            }
            
            //  Make a column-major vector of the training output matrix
            offset = 0
            var y = [Double](repeating: 0.0, count: trainData.size * outputDimension)
            for column in 0..<outputDimension {
                for index in 0..<trainData.size {
                    let outputs = try trainData.getOutput(index)
                    y[offset] = outputs[column]
                    offset += 1
                }
            }
            
            //  Solve the matrix for the parameters Θ (DGELS)
            let jobTChar = "N" as NSString
            var jobT : Int8 = Int8(jobTChar.character(at: 0))          //  not transposed
            var m : Int32 = Int32(trainData.size)
            var n : Int32 = Int32(numColumns)
            var nrhs = Int32(outputDimension)
            var work : [Double] = [0.0]
            var lwork : Int32 = -1        //  Ask for the best size of the work array
            var info : Int32 = 0
            dgels_(&jobT, &m, &n, &nrhs, &A, &m, &y, &m, &work, &lwork, &info)
            if (info != 0 || work[0] < 1) {
                throw LinearRegressionError.matrixSolutionError
            }
            lwork = Int32(work[0])
            work = [Double](repeating: 0.0, count: Int(work[0]))
            dgels_(&jobT, &m, &n, &nrhs, &A, &m, &y, &m, &work, &lwork, &info)
            if (info != 0 || work[0] < 1) {
                throw LinearRegressionError.matrixSolutionError
            }
            
            //  Extract the parameters from the results
            for output in 0..<outputDimension {
                for parameter in 0..<numColumns {
                    Θ[output].append(y[output * trainData.size + parameter])
                }
            }
        }
        
        //  If we have a regularization term, we need to work some of the algebra ourselves
        else {
            //  Get the dimensions of the A matrix
            let nNumPoints = trainData.size
            let N : la_count_t = la_count_t(numColumns)
            let M : la_count_t = la_count_t(nNumPoints)
            
            //  Generate the A Matrix
            var dA = [Double](repeating: 0.0, count: trainData.size * numColumns)
            var offset = 0
            for point in 0..<nNumPoints {
                if (includeBias) {
                    dA[offset] = 1.0
                    offset += 1
                }
                for parameter in 0..<terms.count {
                    do {
                        let inputs = try trainData.getInput(point)
                        dA[offset] = try terms[parameter].getTermValue(inputs)
                    }
                    catch let error {
                        throw error
                    }
                    offset += 1
                }
            }
            
            //  Convert into Linear Algebra objects
            let A = la_matrix_from_double_buffer(dA, M, N,
                                                 N, la_hint_t(LA_NO_HINT), la_attribute_t(LA_DEFAULT_ATTRIBUTES))
            
            //  Calculate A'A
            var AtA = la_matrix_product(la_transpose(A), A)
            
            //  If there is a regularization term, add λI (giving (A'A + λI)
            if let regTerm = regularization {
                let λI = la_scale_with_double(la_identity_matrix(N, la_scalar_type_t(LA_SCALAR_TYPE_DOUBLE), la_attribute_t(LA_DEFAULT_ATTRIBUTES)), regTerm)
                AtA = la_sum(AtA, λI)
            }
            
            //  Iterate through each solution
            var Y = [Double](repeating: 0.0, count: trainData.size)
            for solution in 0..<outputDimension {
                //  Generate the Y vector
                for point in 0..<nNumPoints {
                    let outputs = try trainData.getOutput(point)
                    Y[point] = outputs[solution]
                }
                
                //  Calculate A'Y
                let vY = la_matrix_from_double_buffer(Y, M, 1,
                                                      1, la_hint_t(LA_NO_HINT), la_attribute_t(LA_DEFAULT_ATTRIBUTES))
                let AtY = la_matrix_product(la_transpose(A), vY)
                
                // W = inverse(A'A + λI) * A'Y
                // (A'A + λI)W = A'Y --> of the form Ax = b, we can use the solve function
                let W = la_solve(AtA, AtY)
                
                //  Extract the results back into the learning parameter array
                var parameters = [Double](repeating: 0.0, count: numColumns)
                la_vector_to_double_buffer(&parameters, 1, W)
                Θ[solution] = parameters
            }
        }
    }
    
    open func continueTrainingRegressor(_ trainData: MLRegressionDataSet) throws
    {
        //  Linear regression uses one-batch training (solved analytically)
        throw MachineLearningError.continuationNotSupported
    }
    
    open func predictOne(_ inputs: [Double]) throws ->[Double]
    {
        //  Make sure we are trained
        if (Θ.count < 1) { throw MachineLearningError.notTrained }
        
        //  Get the number of input values needed by the model
        var neededInputs = 0
        for term in terms {
            let termHighest = term.getHighestInputIndex()
            if (termHighest > neededInputs) { neededInputs = termHighest }
        }
        
        //  Validate that the model has the dimension
        if (neededInputs >= inputDimension) {
            throw LinearRegressionError.modelExceedsInputDimension
        }
        
        //  Get the array of term values
        var termValues : [Double] = []
        
        //  If we have a bias term, add it
        if (includeBias) { termValues.append(1.0) }
        
        //  Add each term value
        for term in terms {
            do {
                try termValues.append(term.getTermValue(inputs))
            }
            catch let error {
                throw error
            }
        }
        
        //  Use dot product to multiply the term values by the computed parameters to get each value
        var results : [Double] = []
        var value = 0.0
        for parameters in Θ {
            vDSP_dotprD(termValues, 1, parameters, 1, &value, vDSP_Length(termValues.count))
            results.append(value)
        }
        
        return results
    }
    
    open func predict(_ testData: MLRegressionDataSet) throws
    {
        //  Verify the data set is the right type
        if (testData.dataType != .regression) { throw MachineLearningError.dataNotRegression }
        if (testData.inputDimension != inputDimension) { throw MachineLearningError.dataWrongDimension }
        if (testData.outputDimension != outputDimension) { throw MachineLearningError.dataWrongDimension }
        
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
