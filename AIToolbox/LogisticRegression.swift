//
//  LogisticRegression.swift
//  AIToolbox
//
//  Created by Kevin Coble on 5/13/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation
import Accelerate

///  Class for logistic regression solving/classification
public class LogisticRegression : Regressor, Classifier
{
    let numInputs : Int     //  Does not include bias term
    var numClasses = 2      //  Set after initial training
    let solveType : NonLinearRegressionType
    public var parameters : [Double]
    var initializeFunction : ((trainData: DataSet)->[Double])!
    var classData : ClassificationData!

    ///  Initializer - specify the dimension of the input data
    public init(numInputs : Int, solvingMethod: NonLinearRegressionType) {
        self.numInputs = numInputs
        solveType = solvingMethod
        parameters = [Double](count:numInputs+1, repeatedValue: 0.0)
    }
    
    public func getInputDimension() -> Int
    {
        return numInputs
    }
    public func getOutputDimension() -> Int
    {
        return numClasses
    }
    public func getParameterDimension() -> Int
    {
        return (numInputs + 1) * numClasses * (numClasses - 1) / 2        //  Weights and bias term for each pair of classes
    }
    public func getNumberOfClasses() -> Int        //  May only be valid after training
    {
        return 2     //!!  Currently fixed at a binary set
    }
    public func setParameters(parameters: [Double]) throws
    {
        if (parameters.count < getParameterDimension()) { throw MachineLearningError.NotEnoughData }
        self.parameters = parameters
    }
    ///  Method to set a custom function to initialize the parameters.  If not set, random parameters are used
    public func setCustomInitializer(function: ((trainData: DataSet)->[Double])!)
    {
        initializeFunction = function
    }
    
    public func getParameters() throws -> [Double]
    {
        return parameters
    }
    
    ///  Method to initialize the weights - call before any training other than 'trainClassifier' or 'trainRegressor', which call this
    public func initializeWeights(trainData: DataSet!)
    {
        let numPairs = numClasses * (numClasses - 1) / 2
        if let initFunc = initializeFunction, data = trainData {
            let startWeights = initFunc(trainData: data)
            //  If enough for all parameters, split and send to layers
            if (getParameterDimension() == startWeights.count) {
                parameters = []
                var index = 1 //  First number (if more than 1) goes into the bias weight, then repeat the remaining
                for _ in 0..<numPairs {
                    for _ in 0..<numInputs  {
                        if (index >= startWeights.count) { index = 1 }      //  Wrap if necessary
                        parameters.append(startWeights[index])
                    }
                    parameters.append(startWeights[0])     //  Add the bias term
                }
            }
            else {
                parameters = []
                for _ in 0..<numPairs {
                    for _ in 0..<numInputs  {
                        parameters.append(Gaussian.gaussianRandom(0.0, standardDeviation: 1.0 / Double(numInputs)))    //  input weights - Initialize to a random number to break initial symmetry of the network, scaled to the inputs
                    }
                    parameters.append(Gaussian.gaussianRandom(0.0, standardDeviation:1.0))    //  Bias weight - Initialize to a  random number to break initial symmetry of the network
                }
            }
        }
        else {
            //  No initialization function, set weights to random values
            parameters = []
            for _ in 0..<numPairs {
                for _ in 0..<numInputs  {
                    parameters.append(Gaussian.gaussianRandom(0.0, standardDeviation: 1.0 / Double(numInputs)))    //  input weights - Initialize to a random number to break initial symmetry of the network, scaled to the inputs
                }
                parameters.append(Gaussian.gaussianRandom(0.0, standardDeviation:1.0))    //  Bias weight - Initialize to a  random number to break initial symmetry of the network
            }
        }
    }
    
    public func trainClassifier(trainData: DataSet) throws
    {
        //  Get the class labels
        try trainData.groupClasses()
        classData = trainData.optionalData as? ClassificationData
        numClasses = classData.numClasses
        
        //  Initialize the weights
        initializeWeights(trainData)
        
        //  Use the continue method to train on the initial data
        try trainingClassifier(trainData)
    }
    
    public func continueTrainingClassifier(trainData: DataSet) throws      //  Trains without initializing parameters first
    {
        //  Verify the class labels match the initial set
        try trainData.groupClasses()
        let continueClassData = trainData.optionalData as? ClassificationData
        if (continueClassData!.classCount != classData.classCount) {        //  Must have same number of classes so all pairs can be trained
            throw MachineLearningError.ContinueTrainingClassesNotSame
        }
        for label in classData.foundLabels {
            if (!continueClassData!.foundLabels.contains(label)) {
                throw MachineLearningError.ContinueTrainingClassesNotSame
            }
        }
        
        //  Use the continue method to train on the additional data
        try trainingClassifier(trainData)
    }
    
    func trainingClassifier(trainData: DataSet) throws      //  Continues training on a data set.  Assumes class labels have been set
    {
        //  Create a non-linear solution
        let logit = LogitFunction(numInputs: numInputs)
        let nlr = NonLinearRegression(equation: logit, type: solveType)
        
        //  Train on each pair of classes
        var parameterStart = 0
        for i in 0..<numClasses-1 {
            for j in i+1..<numClasses {
                //  Create a sub-problem data set with just the i and j class data
                let subProblem = DataSet(dataType: .Regression, inputDimension: numInputs, outputDimension: 1)
                do {
                    try subProblem.includeEntryInputs(fromDataSet: trainData, withEntries: classData.classOffsets[i])
                    try subProblem.includeEntryInputs(fromDataSet: trainData, withEntries: classData.classOffsets[j])
                    
                    //  Move parameters into logit for classes being trained
                    for index in 0..<(numInputs+1) {
                        logit.parameters[index] = parameters[parameterStart+index]
                    }
                    
                    //  Set the sub-problem regression values to 0 and 1
                    for index in 0..<classData.classCount[i] {
                        try subProblem.setOutput(index, newOutput: [0.0])
                    }
                    for index in classData.classCount[i]..<subProblem.size {
                        try subProblem.setOutput(index, newOutput: [1.0])
                    }
                    
                    //  Train on the sub-problem
                    nlr.initialStepSize = 0.5
                    try nlr.continueTrainingRegressor(subProblem)       //  We already initialized parameters, so use 'continue' training
                    
                    //  Move parameters back out of the logit
                    for index in 0..<(numInputs+1) {
                        parameters[parameterStart+index] = logit.parameters[index]
                    }
                }
                parameterStart += numInputs + 1
            }
        }
    }
    
    //  Training is done with classification data
    public func trainRegressor(trainData: DataSet) throws
    {
        try trainClassifier(trainData)
    }
    public func continueTrainingRegressor(trainData: DataSet) throws      //  Trains without initializing parameters first
    {
        try continueTrainingClassifier(trainData)
    }
    
    ///  Return the class with the highest probability
    public func classifyOne(inputs: [Double]) throws ->Int
    {
        if (inputs.count != numInputs) { throw DataTypeError.WrongDimensionOnInput }

        //  Create a logit function to do the comparisions
        let logit = LogitFunction(numInputs: numInputs)
        
        //  Allocate vote space for the classification
        var votes = [Int](count: numClasses, repeatedValue: 0)
        
        //  Get the decision for each pair
        var parameterStart = 0
        for i in 0..<numClasses {
            for j in i+1..<numClasses {
                
                //  Move parameters into logit for classes being checked
                for index in 0..<(numInputs+1) {
                    logit.parameters[index] = parameters[parameterStart+index]
                }
                
                //  Get the vote from the logit function
                let result = try logit.getOutputs(inputs)
                if (result[0] < 0.5) {
                    votes[i] += 1
                }
                else {
                    votes[j] += 1
                }

                parameterStart += numInputs + 1
            }
        }
        
        //  Find the class label with the highest votes
        var highestProbabilityClass = 0
        var highestVotes = -1
        for i in 0..<numClasses {
            if (votes[i] > highestVotes) {
                highestVotes = votes[i]
                highestProbabilityClass = classData.foundLabels[i]
            }
        }

        return highestProbabilityClass
    }
    
    ///  Set the class label to the class with the highest probability for each data point
    public func classify(testData: DataSet) throws
    {
        //  Verify the data set is the right type
        if (testData.dataType != .Classification) { throw DataTypeError.InvalidDataType }
        if (testData.inputDimension != numInputs) { throw DataTypeError.WrongDimensionOnInput }
        
        //  Classify each input
        testData.classes = []
        for index in 0..<testData.size {
            try testData.classes!.append(classifyOne(testData.inputs[index]))
        }
    }
    
    ///  Return the probability for each class
    public func predictOne(inputs: [Double]) throws ->[Double]
    {
        //  Verify the data set is the right type
        if (inputs.count != numInputs) { throw DataTypeError.WrongDimensionOnInput }
        
        //  Get the weighted sum
        var weightedSum = 0.0
        vDSP_dotprD(parameters, 1, inputs, 1, &weightedSum, vDSP_Length(numInputs))
        weightedSum += parameters[numInputs]        //  Bias term
        
        //  Run through the sigmoid function
        let probabilityOne = 1.0 / (1.0 + exp(-weightedSum))
        
        return [1.0 -  probabilityOne, probabilityOne]
    }
    
    ///  Set the probablility for each class for each data point
    public func predict(testData: DataSet) throws
    {
        //  Verify the data set is the right type
        if (testData.inputDimension != numInputs) { throw DataTypeError.WrongDimensionOnInput }
        if (testData.outputDimension != getOutputDimension()) { throw DataTypeError.WrongDimensionOnOutput }
        
        //  predict on each input
        testData.outputs = []
        for index in 0..<testData.size {
            try testData.outputs!.append(predictOne(testData.inputs[index]))
        }
    }
}

//  Logit function for a binary class
class LogitFunction : NonLinearEquation {
    let numInputs : Int
    var parameters: [Double] = []
    
    init(numInputs : Int) {
        self.numInputs = numInputs
        parameters = [Double](count:numInputs+1, repeatedValue: 0.0)
    }

    func getInputDimension() -> Int
    {
        return numInputs
    }
    func getOutputDimension() -> Int
    {
        return 1
    }
    func getParameterDimension() -> Int
    {
        return numInputs + 1        //  Bias term
    }
    func setParameters(parameters: [Double]) throws
    {
        if (parameters.count < getParameterDimension()) { throw MachineLearningError.NotEnoughData }
        self.parameters = parameters
    }
    func getOutputs(inputs: [Double]) throws -> [Double]        //  Returns vector outputs sized for outputs
    {
        //  Verify the data set is the right type
        if (inputs.count != numInputs) { throw DataTypeError.WrongDimensionOnInput }

        //  Get the weighted sum
        var weightedSum = 0.0
        vDSP_dotprD(parameters, 1, inputs, 1, &weightedSum, vDSP_Length(numInputs))
        weightedSum += parameters[numInputs]        //  Bias term
        
        //  Run through the sigmoid function
        let probabilityOne = 1.0 / (1.0 + exp(-weightedSum))
        
        return [probabilityOne]
    }
    func getGradient(inputs: [Double]) throws -> [Double]       //  Returns vector gradient with respect to parameters - ∂y/∂w, where y is the output and w is each parameter  (this is multiplied by ∂E/∂y by Nonlinear Regression SGD routine, where E is the error)
    {
        //  Get the output
        let output = try getOutputs(inputs)[0]
        
        //  Get ∂y/∂z (z is weighted sum before sigmoid)
        var dz = output - (output * output)
        
        //  Multiply by ∂z/∂w for each parameter, which is just the input
        var gradient = [Double](count:numInputs, repeatedValue: 0.0)
        vDSP_vsmulD(inputs, 1, &dz, &gradient, 1, vDSP_Length(numInputs))
        gradient.append(dz)        //  Bias term
        
        return gradient
    }
}

