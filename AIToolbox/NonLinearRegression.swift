//
//  NonLinearRegression.swift
//  AIToolbox
//
//  Created by Kevin Coble on 4/15/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation
import Accelerate

///  Enumeration for type of solution attempted for the non-linear rectression
public enum NonLinearRegressionType {
    case ParameterDelta
    case SGD
    case GaussNewton
}

public enum NonLinearRegressionConvergenceType {
    case SmallGradient
    case SmallAverageLoss
    case SmallParameterChange       //  Only option for ParameterDelta
}

public enum NonLinearRegressionError: ErrorType {
    case ConvergenceTypeNotAllowed
    case MatrixSolutionError
}

///  Class for non-linear regression solving
///  Requires an equation that conforms to the protocol 'NonLinearEquation'
public class NonLinearRegression : Regressor
{
    var equation: NonLinearEquation
    let solveType: NonLinearRegressionType
    var batchSize = 0
    var initialStepSize = 0.01
    var stepSizeModifier = 1.0
    var stepSizeChangeAfterIterations = 10000
    var initializeFunction : ((trainData: DataSet)->[Double])!
    public var convergenceType = NonLinearRegressionConvergenceType.SmallGradient
    public var convergenceLimit = 0.0000001     //  Maximum component of gradient at convergence for SGD
    public var iterationLimit = 10000     //  convergence failure after this many iterations
    public var normalizeGradient = false        //  If true, SGD will normalize the gradient vector before multiplying by the step size
    
    public init(equation: NonLinearEquation, type: NonLinearRegressionType)
    {
        self.equation = equation
        solveType = type
    }
    
    ///  Convenience constructor for creating a ParameterDelta solution
    public convenience init(equation: NonLinearEquation, batchSize: Int, initialDelta: Double)
    {
        self.init(equation: equation, type: .ParameterDelta)
        self.batchSize = batchSize
        initialStepSize = initialDelta
        convergenceType = .SmallParameterChange
        convergenceLimit = 0.001
    }
    
    ///  Convenience constructor for creating an SGD solution
    public convenience init(equation: NonLinearEquation, batchSize: Int, initialStepSize: Double, multiplyBy: Double, afterIterations: Int)
    {
        self.init(equation: equation, type: .SGD)
        self.batchSize = batchSize
        self.initialStepSize = initialStepSize
        stepSizeModifier = multiplyBy
        stepSizeChangeAfterIterations = afterIterations
    }
    
    ///  Convenience constructor for creating a Gauss-Newton solution
    public convenience init(equation: NonLinearEquation, batchSize: Int)
    {
        self.init(equation: equation, type: .GaussNewton)
        self.batchSize = batchSize
    }
    
    ///  Method to set a custom function to initialize the parameters.  If not set, random parameters are used
    public func setCustomInitializer(function: ((trainData: DataSet)->[Double])!)
    {
        initializeFunction = function
    }
    
    public func getParameters() throws -> [Double]
    {
        return equation.parameters
    }
    
    ///  Method to set convergence criteria
    public func setConvergence(type: NonLinearRegressionConvergenceType, limit: Double)
    {
        convergenceType = type
        convergenceLimit = limit
    }
    
    public func getInputDimension() -> Int
    {
        return equation.getInputDimension()
    }
    
    public func getOutputDimension() -> Int
    {
        return equation.getOutputDimension()
    }
    
    public func getParameterDimension() -> Int
    {
        return equation.getParameterDimension()
    }
    
    public func setParameters(parameters: [Double]) throws
    {
        try equation.setParameters(parameters)
    }

    
    public func trainRegressor(trainData: DataSet) throws
    {
        //  Validate the training data
        if (trainData.dataType != .Regression) { throw MachineLearningError.DataNotRegression }
        if (trainData.inputDimension != equation.getInputDimension()) { throw MachineLearningError.DataWrongDimension }
        if (trainData.outputDimension != equation.getOutputDimension()) { throw MachineLearningError.DataWrongDimension }
        
        //  If batch size is zero, size it for the entire data set
        if (batchSize <= 0) {batchSize = trainData.size }
        
        //  Initialize the parameters
        let numParameters = equation.getParameterDimension()
        if let initFunc = initializeFunction {
            let initParameters = initFunc(trainData: trainData)
            if (initParameters.count != numParameters) { throw MachineLearningError.InitializationError }
            equation.parameters = initParameters
        }
        else {
            var initParameters: [Double] = []
            for _ in 0..<numParameters {
                initParameters.append(Double(arc4random()) / Double(UInt32.max) - 0.5)
            }
            equation.parameters = initParameters
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
        //  Validate the training data
        if (trainData.dataType != .Regression) { throw MachineLearningError.DataNotRegression }
        if (trainData.inputDimension != equation.getInputDimension()) { throw MachineLearningError.DataWrongDimension }
        if (trainData.outputDimension != equation.getOutputDimension()) { throw MachineLearningError.DataWrongDimension }
        
        //  If batch size is zero, size it for the entire data set
        if (batchSize <= 0) {batchSize = trainData.size }

        //  Use the specified method to converge the parameters
        do {
            switch (solveType) {
            case .ParameterDelta:
                try trainParameterDelta(trainData)
                
            case .SGD:
                try trainSGD(trainData)
                
            case .GaussNewton:
                try trainGaussNewton(trainData)
            }
        }
        catch let error {
            throw error
        }
    }
    
    public func trainParameterDelta(trainData: DataSet) throws
    {
        //  Make sure we have the right convergence type
        if (convergenceType != .SmallParameterChange) { throw NonLinearRegressionError.ConvergenceTypeNotAllowed }
        
        //  Set up the batch indices
        var tOrder : [(index : Int, random : Double)] = []
        for i in 0..<trainData.size { tOrder.append((index : i, random : 0.0)) }
        var batchPointIndex = trainData.size    //  Start with a new random set
        if (batchSize == trainData.size) {batchPointIndex = 0 }
        
        //  Get settings needed
        let numParameters = equation.getParameterDimension()
        let numOutputs = equation.getOutputDimension()
        let parametersPerOutput = numParameters / numOutputs
        
        var absoluteParameterChanges = [Double](count: numParameters, repeatedValue: 0.0)
        var modifierStart = initialStepSize
        var iteration = 0
        while (iteration < iterationLimit) {
            iteration += 1
            
            //  See if we have enough data for the batch
            if (batchSize != trainData.size) {
                if (batchPointIndex + batchSize >= trainData.size) {
                    //  Get a new random set of points
                    //  Get a random order for the input points
                    for i in 0..<trainData.size {
                        tOrder[i].random = drand48()
                    }
                    tOrder.sortInPlace{$0.random < $1.random}
                    batchPointIndex = 0
                }
            }
            else {
                batchPointIndex = 0     //  Full data set batch, restart at beginning
            }
            
            //  Remember the parameters
            let previousParameters = equation.parameters
            
            var totalChange = [Double](count: numParameters, repeatedValue: 0.0)
            
            //  Get the parameter modifications for this batch
            do {
                for _ in 0..<batchSize {
                    //  Get the point
                    let point = tOrder[batchPointIndex].index
                    batchPointIndex += 1
                    
                    //  Get value of function with current parameter value
                    let previousResults = try equation.getOutputs(trainData.inputs[point])

                    //  Update the parameters for each output
                    for output in 0..<numOutputs {
                        
                        //  If this output is already matching close, skip processing
                        let outputValue = trainData.outputs![point][output]
                        if (fabs(previousResults[output] - outputValue) < 0.0000000001) {continue}
                        
                        //  Process each parameter
                        for param in 0..<parametersPerOutput {
                            let parameter = output * parametersPerOutput + param    //  Parameter index
                            
                            //  Modify the parameters
                            var modifier = modifierStart
                            equation.parameters[parameter] += modifier
                            
                            //  Get the new value
                            let updatedResult = try equation.getOutputs(trainData.inputs[point])[output]
                            let difference = updatedResult - previousResults[output]
                            if (fabs(difference) < 0.0000000001) {continue}         //  If this parameter change didn't affect things, skip
                            
                            //  Put the parameter back to where we started for the next parameter difference check
                            equation.parameters[parameter] = previousParameters[parameter]
                            
                            //  Calculate the parameter modification that would result in exact value assuming constant linear gradient
                            modifier *= (outputValue - previousResults[output]) / difference
                            totalChange[parameter] += modifier
                        }
                    }
                }
                
                //  Change the parameters by the average change for the batch
                var scale = 1.0 / Double(batchSize)
                vDSP_vsmulD(totalChange, 1, &scale, &totalChange, 1, vDSP_Length(numParameters))
                vDSP_vaddD(totalChange, 1, equation.parameters, 1, &equation.parameters, 1, vDSP_Length(numParameters))
            }
            catch let error {
                throw error
            }
            
            //  Check for convergence
            vDSP_vabsD(totalChange, 1, &absoluteParameterChanges, 1, vDSP_Length(numParameters))
            var totalParameterChange = 0.0
            vDSP_sveD(absoluteParameterChanges, 1, &totalParameterChange, vDSP_Length(numParameters))
            if (totalParameterChange < convergenceLimit) {
                return
            }
            
            //  Lower delta each iteration
            modifierStart *= 0.5
        }
    }
   
    public func trainSGD(trainData: DataSet) throws
    {
        //  Set up the batch indices
        var tOrder : [(index : Int, random : Double)] = []
        for i in 0..<trainData.size { tOrder.append((index : i, random : 0.0)) }
        var batchPointIndex = trainData.size    //  Start with a new random set
        if (batchSize == trainData.size) {batchPointIndex = 0 }
        
        //  Get settings needed
        let numParameters = equation.getParameterDimension()
        let numOutputs = equation.getOutputDimension()
        let parametersPerOutput = numParameters / numOutputs
        
        var stepSizeChangeIteration = stepSizeChangeAfterIterations
        var stepSize = initialStepSize
        var totalLoss: Double
        var lossIncrementCount = 0
        var lastTotalLoss = Double.infinity
        var dotProduct = 0.0
        var absoluteValGradient = [Double](count: numParameters, repeatedValue: 0.0)
        
        var iteration = 0
        while (iteration < iterationLimit) {
            iteration += 1
            stepSizeChangeIteration -= 1
            if (stepSizeChangeIteration == 0) {
                stepSize *= stepSizeModifier
                stepSizeChangeIteration = stepSizeChangeAfterIterations
            }
            
            //  See if we have enough data for the batch
            if (batchSize != trainData.size) {
                if (batchPointIndex + batchSize >= trainData.size) {
                    //  Get a new random set of points
                    //  Get a random order for the input points
                    for i in 0..<trainData.size {
                        tOrder[i].random = drand48()
                    }
                    tOrder.sortInPlace{$0.random < $1.random}
                    batchPointIndex = 0
                }
            }
            else {
                batchPointIndex = 0     //  Full data set batch, restart at beginning
            }
            
            //  Get the average gradient
            totalLoss = 0.0
            var averageGradient = [Double](count: numParameters, repeatedValue: 0.0)
            do {
                for _ in 0..<batchSize {
                    //  Get the point
                    let point = tOrder[batchPointIndex].index
                    batchPointIndex += 1
                    //  gradient with respect to least-squares loss function is 2 * (f(x) - y) * ∂f(x)/∂x
                    var values = try equation.getOutputs(trainData.inputs[point])       //  Get f(x)
                    vDSP_vsubD(trainData.outputs![point], 1, values, 1, &values, 1, vDSP_Length(numOutputs))    //  Subtract y
                    vDSP_dotprD(values, 1, values, 1, &dotProduct, vDSP_Length(numOutputs))     //  Loss term is square of difference - use dot product for quick tally of squares
                    totalLoss += dotProduct
                    if (parametersPerOutput > 1) {       //  Extend each difference just calculated to the parameter subset it belongs to
                        var extendedValues : [Double] = []
                        for value in values { extendedValues += [Double](count: parametersPerOutput, repeatedValue: value) }
                        values = extendedValues
                    }
                    var gradient = try equation.getGradient(trainData.inputs[point])        //  get ∂f(x)/∂x
                    vDSP_vmulD(gradient, 1, values, 1, &gradient, 1, vDSP_Length(numParameters))        //  multiply (f(x) - y) by ∂f(x)/∂x
                    vDSP_vaddD(gradient, 1, averageGradient, 1, &averageGradient, 1, vDSP_Length(numParameters))        //  Add for average gradient
                }
                var scale = 2.0 / Double(batchSize)        //  2 from derivitive above - cheaper to do it here
                vDSP_vsmulD(averageGradient, 1, &scale, &averageGradient, 1, vDSP_Length(numParameters))        //  Finish the average
            }
            catch let error {
                throw error
            }
            
            //  If total loss has not gone down in 4 iterations, cut step size in half
            //  If total loss has not gone down in 13 iterations (three step size changes) and we are using random initialization, re-initialize to a new set of parameters
            if (totalLoss != Double.infinity && totalLoss != Double.NaN && totalLoss < lastTotalLoss) {
                lossIncrementCount = 0
            }
            else {
                lossIncrementCount += 1
                if ((lossIncrementCount % 4) == 0) { stepSize *= 0.5 }
                if (lossIncrementCount >= 13 && initializeFunction == nil) {
                    var initParameters: [Double] = []
                    for _ in 0..<numParameters {
                        initParameters.append(Double(arc4random()) / Double(UInt32.max) - 0.5)
                    }
                    equation.parameters = initParameters
                    lossIncrementCount = 0
                    lastTotalLoss = Double.infinity
                }
            }
            lastTotalLoss = totalLoss
            
            //  Check for convergence
            switch (convergenceType) {
            case .SmallAverageLoss:
                //  See if the average loss is less than the convergence limit
                totalLoss /= Double(batchSize)     // Get average loss
                if (totalLoss < convergenceLimit) {
                    return
                }
            case .SmallGradient:
                //  See if the maximum derivitive is less than the convergence limit
                vDSP_vabsD(averageGradient, 1, &absoluteValGradient, 1, vDSP_Length(numParameters))
                var maxDerivitive = convergenceLimit + 1.0
                vDSP_maxvD(absoluteValGradient, 1, &maxDerivitive, vDSP_Length(numParameters))
                if (maxDerivitive < convergenceLimit) {
                    return
                }
            case .SmallParameterChange:
                vDSP_vabsD(averageGradient, 1, &absoluteValGradient, 1, vDSP_Length(numParameters))
                var totalParameterChange = 0.0
                vDSP_sveD(absoluteValGradient, 1, &totalParameterChange, vDSP_Length(numParameters))
                if (totalParameterChange < convergenceLimit) {
                    return
                }
            }
            
            //  If specified, normalize the gradient to a unit vector
            if (normalizeGradient) {
                var normSquared = 1.0
                vDSP_dotprD(averageGradient, 1, averageGradient, 1, &normSquared, vDSP_Length(numParameters))
                if (normSquared > 0.00001) {
                    var scale = 1.0 / sqrt(normSquared)
                    vDSP_vsmulD(averageGradient, 1, &scale, &averageGradient, 1, vDSP_Length(numParameters))
                }
            }
        
            //  Move the parameters by the step times the average gradient
            vDSP_vsmulD(averageGradient, 1, &stepSize, &averageGradient, 1, vDSP_Length(numParameters))
            vDSP_vsubD(averageGradient, 1, equation.parameters, 1, &equation.parameters, 1, vDSP_Length(numParameters))
        }
        throw MachineLearningError.DidNotConverge
    }
    
    public func trainGaussNewton(trainData: DataSet) throws
    {
        //  Set up the batch indices
        var tOrder : [(index : Int, random : Double)] = []
        for i in 0..<trainData.size { tOrder.append((index : i, random : 0.0)) }
        var batchPointIndex = trainData.size    //  Start with a new random set
        if (batchSize == trainData.size) {batchPointIndex = 0 }
        
        //  Get settings needed
        let numParameters = equation.getParameterDimension()
        let numOutputs = equation.getOutputDimension()
        let parametersPerOutput = numParameters / numOutputs
        
        var iteration = 0
        var J = [Double](count: batchSize * numParameters, repeatedValue: 0.0)
        var r = [Double](count: batchSize * numOutputs, repeatedValue: 0.0)
        var delta = [Double](count: numParameters, repeatedValue: 0.0)
        while (iteration < iterationLimit) {
            iteration += 1
            
            //  See if we have enough data for the batch
            if (batchSize != trainData.size) {
                if (batchPointIndex + batchSize >= trainData.size) {
                    //  Get a new random set of points
                    //  Get a random order for the input points
                    for i in 0..<trainData.size {
                        tOrder[i].random = drand48()
                    }
                    tOrder.sortInPlace{$0.random < $1.random}
                    batchPointIndex = 0
                }
            }
            else {
                batchPointIndex = 0     //  Full data set batch, restart at beginning
            }
        
            //  Calculate the residual vectors and the Jacobian matrix (column major order)
            var totalLoss = 0.0
            var maxDerivitive = 0.0
            do {
                for pointIndex in 0..<batchSize {
                    //  Get the point
                    let point = tOrder[batchPointIndex].index
                    batchPointIndex += 1
                    
                    //  Get the residual
                    let output = try equation.getOutputs(trainData.inputs[point])
                    for outputIndex in 0..<numOutputs {
                        let residual = output[outputIndex] - trainData.outputs![point][outputIndex]
                        r[outputIndex * batchSize + pointIndex] = residual
                        if (convergenceType == .SmallAverageLoss) {
                            totalLoss += fabs(residual)
                        }
                    }
                    
                    //  Get gradient at each point for the Jacobian
                    let gradient = try equation.getGradient(trainData.inputs[point])
                    for parameter in 0..<numParameters {
                        J[(parameter * batchSize) + pointIndex] = gradient[parameter]
                        if (convergenceType == .SmallGradient) {
                            if (gradient[parameter] > maxDerivitive) { maxDerivitive = gradient[parameter] }
                        }
                    }
                }
            }
            catch let error {
                throw error
            }
            
            //  If the convergence type is a small loss, check if we are done
            if (convergenceType == .SmallAverageLoss) {
                totalLoss /= Double(batchSize)
                if (totalLoss < convergenceLimit) { return }
            }
            
            //  If the convergence type is a small gradient, check if we are done
            if (convergenceType == .SmallGradient) {
                if (maxDerivitive < convergenceLimit) { return }
            }
            
            //  Process each output seperately
            for outputIndex in 0..<numOutputs {
                //  Solve J'Jp = J'r for p - the parameter change, using LAPACK's dgels function
                let jobTChar = "N" as NSString
                var jobT : Int8 = Int8(jobTChar.characterAtIndex(0))          //  not transposed
                var m : Int32 = Int32(batchSize)
                var n : Int32 = Int32(parametersPerOutput)
                var nrhs = Int32(1)
                var work : [Double] = [0.0]
                var lwork : Int32 = -1        //  Ask for the best size of the work array
                var info : Int32 = 0
                let jacobianOffset = batchSize * outputIndex * parametersPerOutput      //  Offset to start of Jacobian for this output
                let residualOffset = batchSize * outputIndex      //  Offset to start of residual vector for this output
                dgels_(&jobT, &m, &n, &nrhs, &J[jacobianOffset], &m, &r[residualOffset], &m, &work, &lwork, &info)
                if (info != 0 || work[0] < 1) {
                    throw NonLinearRegressionError.MatrixSolutionError
                }
                lwork = Int32(work[0])
                work = [Double](count: Int(work[0]), repeatedValue: 0.0)
                dgels_(&jobT, &m, &n, &nrhs, &J[jacobianOffset], &m, &r[residualOffset], &m, &work, &lwork, &info)
                if (info != 0 || work[0] < 1) {
                    throw NonLinearRegressionError.MatrixSolutionError
                }
                
                //  Extract the parameter changes
                for parameter in 0..<parametersPerOutput {
                    delta[outputIndex * parametersPerOutput + parameter] = r[parameter]
                }
            }
            
            //  Subtract the parameter change from the parameters
            vDSP_vsubD(delta, 1, equation.parameters, 1, &equation.parameters, 1, vDSP_Length(numParameters))
            
            //  If the convergence criteria is for a small parameter change, check now
            if (convergenceType == .SmallParameterChange) {
                var sum: Double = 0.0;
                vDSP_vswsumD(r, 1, &sum, 1, 1, vDSP_Length(numParameters));
                if (sum < convergenceLimit) { return }
            }
        }
        throw MachineLearningError.DidNotConverge
    }

    
    public func predictOne(inputs: [Double]) throws ->[Double]
    {
        if (inputs.count != equation.getInputDimension()) { throw MachineLearningError.DataWrongDimension }
        if (equation.parameters.count != equation.getParameterDimension()) { throw MachineLearningError.NotTrained }
        
        do {
            return try equation.getOutputs(inputs)
        }
        catch let error {
            throw error
        }
    }
    
    public func predict(testData: DataSet) throws
    {
        //  Verify the data set is the right type
        if (testData.dataType != .Regression) { throw MachineLearningError.DataNotRegression }
        if (testData.inputDimension != equation.getInputDimension()) { throw MachineLearningError.DataWrongDimension }
        
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