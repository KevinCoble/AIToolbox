//
//  File.swift
//  AIToolbox
//
//  Created by Kevin Coble on 4/4/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation
import Accelerate


public enum LinearRegressionError: ErrorType {
    case ModelExceedsInputDimension
    case MatrixSolutionError
    case NegativeInLogOrPower
    case DivideByZero
}


///  Function for subterm
public enum SubtermFunction { case
    None,
    NaturalExponent,
    Sine,
    Cosine,
    Log,
    Power       //  uses power value of term raised to input
}

///  Struct for a piece of a term of the linear equation being fitted
///  Subterms can be combined -->  a₁²a₂² is two subterms, a₁² mulitplied by a₂² in terms
public struct LinearRegressionSubTerm
{
    let inputIndex : Int
    public var power = 1.0
    public var function = SubtermFunction.None
    public var divide = false      //  If true this term divides the previous subterm (ignored on first subterm), else multiply
    
    public init(withInput: Int)
    {
        inputIndex = withInput
    }
    
    public func getSubTermValue(withInputs: [Double]) throws -> Double
    {
        var result = withInputs[inputIndex]
        if (power != 1.0) {
            let powerResult = pow(result, power)        //  Allow negatives if power is integer - if not, nan will result
            if (result < 0.0 && powerResult.isNaN) { throw LinearRegressionError.NegativeInLogOrPower }
            result = powerResult
        }
        switch (function) {
        case .None:
            break
        case .NaturalExponent:
            result = exp(result)
        case .Sine:
            result = sin(result)
        case .Cosine:
            result = cos(result)
        case .Log:
            if (result < 0.0) { throw LinearRegressionError.NegativeInLogOrPower }
            result = log(result)
        case .Power:       //  uses power value of term raised to input
            if (power < 0.0) { throw LinearRegressionError.NegativeInLogOrPower }
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
    
    public mutating func addSubTerm(subTerm: LinearRegressionSubTerm)
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
    
    public func getTermValue(withInputs: [Double]) throws -> Double
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
                        throw LinearRegressionError.DivideByZero
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
public class LinearRegressionModel : Regressor
{
    public private(set) var inputDimension : Int
    public private(set) var outputDimension : Int
    var terms : [LinearRegressionTerm] = []
    public var includeBias = true
    public var regularization : Double?
    public var Θ : [[Double]] = []
    
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
    
    public func addTerm(newTerm: LinearRegressionTerm)
    {
        terms.append(newTerm)
    }
    
    public func getInputDimension() -> Int
    {
        return inputDimension
    }
    
    public func getOutputDimension() -> Int
    {
        return outputDimension
    }
    
    public func getParameterDimension() -> Int
    {
        var numParameters = terms.count
        if (includeBias) { numParameters += 1 }
        numParameters *= outputDimension
        return numParameters
    }
    
    public func setParameters(parameters: [Double]) throws
    {
        if (parameters.count < getParameterDimension()) { throw MachineLearningError.NotEnoughData }
        
        var numParameters = terms.count
        if (includeBias) { numParameters += 1 }
        var offset = 0
        if (Θ.count < outputDimension) { Θ = [[Double]](count: outputDimension, repeatedValue: []) }
        for index in 0..<outputDimension {
            Θ[index] = Array(parameters[offset..<(offset+numParameters)])
            offset += numParameters
        }
    }
    
    public func setCustomInitializer(function: ((trainData: DataSet)->[Double])!) {
        //  Ignore, as Linear Regression doesn't use an initialization
    }
    
    public func getParameters() throws -> [Double]
    {
        var parameters : [Double] = []
        for index in 0..<outputDimension {
            parameters += Θ[index]
        }
        return parameters
    }

    public func trainRegressor(trainData: DataSet) throws
    {
        //  Verify that the data is regression data
        if (trainData.dataType != DataSetType.Regression) { throw MachineLearningError.DataNotRegression }
        
        //  Get the number of input values needed by the model
        var neededInputs = 0
        for term in terms {
            let termHighest = term.getHighestInputIndex()
            if (termHighest > neededInputs) { neededInputs = termHighest }
        }
        
        //  Validate that the model has the dimension
        if (neededInputs >= inputDimension) {
            throw LinearRegressionError.ModelExceedsInputDimension
        }
        
        //  Validate the data size
        if (trainData.inputDimension != inputDimension || trainData.outputDimension != outputDimension) {
            throw MachineLearningError.DataWrongDimension
        }
        
        //  Get the number of terms in the matrix (columns)
        let numColumns = getParameterDimension()
        
        //  Make sure we have enough data for a solution (at least 1 per term)
        if (trainData.size < numColumns) {
            throw MachineLearningError.NotEnoughData
        }
        
        //  Allocate the parameter array
        Θ = [[Double]](count: outputDimension, repeatedValue: [])
        
        //  We can use lapack dgels if no regularization term
        if (regularization == nil) {
       
            //  Make a column-major matrix of the data, with any bias term
            var A = [Double](count: trainData.size * numColumns, repeatedValue: 0.0)
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
                        A[offset] = try terms[parameter].getTermValue(trainData.inputs[index])
                    }
                    catch let error {
                        throw error
                    }
                    offset += 1
                }
            }
            
            //  Make a column-major vector of the training output matrix
            offset = 0
            var y = [Double](count: trainData.size * outputDimension, repeatedValue: 0.0)
            for column in 0..<outputDimension {
                for index in 0..<trainData.size {
                    y[offset] = trainData.outputs![index][column]
                    offset += 1
                }
            }
            
            //  Solve the matrix for the parameters Θ (DGELS)
            let jobTChar = "N" as NSString
            var jobT : Int8 = Int8(jobTChar.characterAtIndex(0))          //  not transposed
            var m : Int32 = Int32(trainData.size)
            var n : Int32 = Int32(numColumns)
            var nrhs = Int32(outputDimension)
            var work : [Double] = [0.0]
            var lwork : Int32 = -1        //  Ask for the best size of the work array
            var info : Int32 = 0
            dgels_(&jobT, &m, &n, &nrhs, &A, &m, &y, &m, &work, &lwork, &info)
            if (info != 0 || work[0] < 1) {
                throw LinearRegressionError.MatrixSolutionError
            }
            lwork = Int32(work[0])
            work = [Double](count: Int(work[0]), repeatedValue: 0.0)
            dgels_(&jobT, &m, &n, &nrhs, &A, &m, &y, &m, &work, &lwork, &info)
            if (info != 0 || work[0] < 1) {
                throw LinearRegressionError.MatrixSolutionError
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
            var dA = [Double](count: trainData.size * numColumns, repeatedValue: 0.0)
            var offset = 0
            for point in 0..<nNumPoints {
                if (includeBias) {
                    dA[offset] = 1.0
                    offset += 1
                }
                for parameter in 0..<terms.count {
                    do {
                        dA[offset] = try terms[parameter].getTermValue(trainData.inputs[point])
                    }
                    catch let error {
                        throw error
                    }
                    offset += 1
                }
            }
            
            //  Convert into Linear Algebra objects`
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
            var Y = [Double](count: trainData.size, repeatedValue: 0.0)
            for solution in 0..<outputDimension {
                //  Generate the Y vector
                for point in 0..<nNumPoints {
                    Y[point] = trainData.outputs![point][solution]
                }
                
                //  Calculate A'Y
                let vY = la_matrix_from_double_buffer(Y, M, 1,
                                                      1, la_hint_t(LA_NO_HINT), la_attribute_t(LA_DEFAULT_ATTRIBUTES))
                let AtY = la_matrix_product(la_transpose(A), vY)
                
                // W = inverse(A'A + λI) * A'Y
                // (A'A + λI)W = A'Y --> of the form Ax = b, we can use the solve function
                let W = la_solve(AtA, AtY)
                
                //  Extract the results back into the learning parameter array
                var parameters = [Double](count: numColumns, repeatedValue: 0.0)
                la_vector_to_double_buffer(&parameters, 1, W)
                Θ[solution] = parameters
            }
        }
    }
    
    public func continueTrainingRegressor(trainData: DataSet) throws
    {
        //  Linear regression uses one-batch training (solved analytically)
        throw MachineLearningError.ContinuationNotSupported
    }
    
    public func predictOne(inputs: [Double]) throws ->[Double]
    {
        //  Make sure we are trained
        if (Θ.count < 1) { throw MachineLearningError.NotTrained }
        
        //  Get the number of input values needed by the model
        var neededInputs = 0
        for term in terms {
            let termHighest = term.getHighestInputIndex()
            if (termHighest > neededInputs) { neededInputs = termHighest }
        }
        
        //  Validate that the model has the dimension
        if (neededInputs >= inputDimension) {
            throw LinearRegressionError.ModelExceedsInputDimension
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
    
    public func predict(testData: DataSet) throws
    {
        //  Verify the data set is the right type
        if (testData.dataType != .Regression) { throw MachineLearningError.DataNotRegression }
        if (testData.inputDimension != inputDimension) { throw MachineLearningError.DataWrongDimension }
        if (testData.outputDimension != outputDimension) { throw MachineLearningError.DataWrongDimension }
        
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