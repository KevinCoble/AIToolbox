//
//  LSTMNeuralNetwork.swift
//  AIToolbox
//
//  Created by Kevin Coble on 5/20/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation
import Accelerate


final class LSTMNeuralNode {
    //  Activation function
    let activation : NeuralActivationFunction
    let numInputs : Int
    let numFeedback : Int
    
    //  Weights
    let numWeights : Int        //  This includes weights from inputs and from feedback for input, forget, cell, and output
    var inputGateInputWeights : [Double]
    var inputGateFeedbackWeights : [Double]
    var forgetGateInputWeights : [Double]
    var forgetGateFeedbackWeights : [Double]
    var memoryCellInputWeights : [Double]
    var memoryCellFeedbackWeights : [Double]
    var outputGateInputWeights : [Double]
    var outputGateFeedbackWeights : [Double]

    var lastOutput : Double //  Last result calculated
    var delta : Double      //  Difference in expected output to calculated result - weighted sum from all nodes this node outputs too

    ///  Create the LSTM neural network node with a set activation function
    init(numInputs : Int, numFeedbacks : Int,  activationFunction: NeuralActivationFunction)
    {
        activation = activationFunction
        self.numInputs = numInputs + 1        //  Add one weight for the bias term
        self.numFeedback = numFeedbacks
        
        //  Weights
        numWeights = (self.numInputs + self.numFeedback) * 4  //  input, forget, cell and output all have weights
        inputGateInputWeights = []
        inputGateFeedbackWeights = []
        forgetGateInputWeights = []
        forgetGateFeedbackWeights = []
        memoryCellInputWeights = []
        memoryCellFeedbackWeights = []
        outputGateInputWeights = []
        outputGateFeedbackWeights = []
        
        lastOutput = 0.0
        delta = 0.0
    }
    
    //  Initialize the weights
    func initWeights(startWeights: [Double]!)
    {
        if let startWeights = startWeights {
            if (startWeights.count == 1) {
                inputGateInputWeights = [Double](count: numInputs, repeatedValue: startWeights[0])
                inputGateFeedbackWeights = [Double](count: numFeedback, repeatedValue: startWeights[0])
                forgetGateInputWeights = [Double](count: numInputs, repeatedValue: startWeights[0])
                forgetGateFeedbackWeights = [Double](count: numFeedback, repeatedValue: startWeights[0])
                memoryCellInputWeights = [Double](count: numInputs, repeatedValue: startWeights[0])
                memoryCellFeedbackWeights = [Double](count: numFeedback, repeatedValue: startWeights[0])
                outputGateInputWeights = [Double](count: numInputs, repeatedValue: startWeights[0])
                outputGateFeedbackWeights = [Double](count: numFeedback, repeatedValue: startWeights[0])
            }
            else if (startWeights.count == (numInputs+numFeedback) * 4) {
                //  Full weight array, just split into the eight weight arrays
                var index = 0
                inputGateInputWeights = Array(startWeights[index..<index+numInputs])
                index += numInputs
                inputGateFeedbackWeights = Array(startWeights[index..<index+numFeedback])
                index += numFeedback
                forgetGateInputWeights = Array(startWeights[index..<index+numInputs])
                index += numInputs
                forgetGateFeedbackWeights = Array(startWeights[index..<index+numFeedback])
                index += numFeedback
                memoryCellInputWeights = Array(startWeights[index..<index+numInputs])
                index += numInputs
                memoryCellFeedbackWeights = Array(startWeights[index..<index+numFeedback])
                index += numFeedback
                outputGateInputWeights = Array(startWeights[index..<index+numInputs])
                index += numInputs
                outputGateFeedbackWeights = Array(startWeights[index..<index+numFeedback])
                index += numFeedback
            }
            else {
                //  Get the weights and bias start indices
                let numValues = startWeights.count
                var inputStart : Int
                var forgetStart : Int
                var cellStart : Int
                var outputStart : Int
                var sectionLength : Int
                if ((numValues % 4) == 0) {
                    //  Evenly divisible by 4, pass each quarter
                    sectionLength = numValues / 4
                    inputStart = 0
                    forgetStart = sectionLength
                    cellStart = sectionLength * 2
                    outputStart = sectionLength * 3
                }
                else {
                    //  Use the values for all sections
                    inputStart = 0
                    forgetStart = 0
                    cellStart = 0
                    outputStart = 0
                    sectionLength = numValues
                }
                
                inputGateInputWeights = []
                var index = inputStart //  Last number (if more than 1) goes into the bias weight, then repeat the initial
                for _ in 0..<numInputs-1  {
                    if (index >= sectionLength-1) { index = inputStart }      //  Wrap if necessary
                    inputGateInputWeights.append(startWeights[index])
                    index += 1
                }
                inputGateInputWeights.append(startWeights[inputStart + sectionLength -  1])     //  Add the bias term
                
                inputGateFeedbackWeights = []
                for _ in 0..<numFeedback  {
                    if (index >= sectionLength-1) { index = inputStart }      //  Wrap if necessary
                    inputGateFeedbackWeights.append(startWeights[index])
                    index += 1
                }
                
                index = forgetStart
                forgetGateInputWeights = []
                for _ in 0..<numInputs-1  {
                    if (index >= sectionLength-1) { index = forgetStart }      //  Wrap if necessary
                    inputGateInputWeights.append(startWeights[index])
                    index += 1
                }
                forgetGateInputWeights.append(startWeights[forgetStart + sectionLength -  1])     //  Add the bias term
                
                forgetGateFeedbackWeights = []
                for _ in 0..<numFeedback  {
                    if (index >= sectionLength-1) { index = forgetStart }      //  Wrap if necessary
                    forgetGateFeedbackWeights.append(startWeights[index])
                    index += 1
                }
                
                index = cellStart
                memoryCellInputWeights = []
                for _ in 0..<numInputs-1  {
                    if (index >= sectionLength-1) { index = cellStart }      //  Wrap if necessary
                    memoryCellInputWeights.append(startWeights[index])
                    index += 1
                }
                memoryCellInputWeights.append(startWeights[cellStart + sectionLength -  1])     //  Add the bias term
                
                memoryCellFeedbackWeights = []
                for _ in 0..<numFeedback  {
                    if (index >= sectionLength-1) { index = cellStart }      //  Wrap if necessary
                    memoryCellFeedbackWeights.append(startWeights[index])
                    index += 1
                }
                
                index = outputStart
                outputGateInputWeights = []
                for _ in 0..<numInputs-1  {
                    if (index >= sectionLength-1) { index = outputStart }      //  Wrap if necessary
                    outputGateInputWeights.append(startWeights[index])
                    index += 1
                }
                outputGateInputWeights.append(startWeights[outputStart + sectionLength -  1])     //  Add the bias term
                
                outputGateFeedbackWeights = []
                for _ in 0..<numFeedback  {
                    if (index >= sectionLength-1) { index = outputStart }      //  Wrap if necessary
                    outputGateFeedbackWeights.append(startWeights[index])
                    index += 1
                }
            }
        }
        else {
            inputGateInputWeights = []
            for _ in 0..<numInputs-1  {
                inputGateInputWeights.append(Gaussian.gaussianRandom(0.0, standardDeviation: 1.0 / Double(numInputs-1)))    //  input weights - Initialize to a random number to break initial symmetry of the network, scaled to the inputs
            }
            inputGateInputWeights.append(Gaussian.gaussianRandom(0.0, standardDeviation:1.0))    //  Bias weight - Initialize to a  random number to break initial symmetry of the network
            
            inputGateFeedbackWeights = []
            for _ in 0..<numFeedback  {
                inputGateFeedbackWeights.append(Gaussian.gaussianRandom(0.0, standardDeviation: 1.0 / Double(numFeedback)))    //  feedback weights - Initialize to a random number to break initial symmetry of the network, scaled to the inputs
            }

            forgetGateInputWeights = []
            for _ in 0..<numInputs-1  {
                forgetGateInputWeights.append(Gaussian.gaussianRandom(0.0, standardDeviation: 1.0 / Double(numInputs-1)))    //  input weights - Initialize to a random number to break initial symmetry of the network, scaled to the inputs
            }
            forgetGateInputWeights.append(Gaussian.gaussianRandom(0.0, standardDeviation:1.0))    //  Bias weight - Initialize to a  random number to break initial symmetry of the network
            
            forgetGateFeedbackWeights = []
            for _ in 0..<numFeedback  {
                forgetGateFeedbackWeights.append(Gaussian.gaussianRandom(0.0, standardDeviation: 1.0 / Double(numFeedback)))    //  feedback weights - Initialize to a random number to break initial symmetry of the network, scaled to the inputs
            }
            
            memoryCellInputWeights = []
            for _ in 0..<numInputs-1  {
                memoryCellInputWeights.append(Gaussian.gaussianRandom(0.0, standardDeviation: 1.0 / Double(numInputs-1)))    //  input weights - Initialize to a random number to break initial symmetry of the network, scaled to the inputs
            }
            memoryCellInputWeights.append(Gaussian.gaussianRandom(0.0, standardDeviation:1.0))    //  Bias weight - Initialize to a  random number to break initial symmetry of the network
            
            memoryCellFeedbackWeights = []
            for _ in 0..<numFeedback  {
                memoryCellFeedbackWeights.append(Gaussian.gaussianRandom(0.0, standardDeviation: 1.0 / Double(numFeedback)))    //  feedback weights - Initialize to a random number to break initial symmetry of the network, scaled to the inputs
            }
            
            outputGateInputWeights = []
            for _ in 0..<numInputs-1  {
                outputGateInputWeights.append(Gaussian.gaussianRandom(0.0, standardDeviation: 1.0 / Double(numInputs-1)))    //  input weights - Initialize to a random number to break initial symmetry of the network, scaled to the inputs
            }
            outputGateInputWeights.append(Gaussian.gaussianRandom(0.0, standardDeviation:1.0))    //  Bias weight - Initialize to a  random number to break initial symmetry of the network
            
            outputGateFeedbackWeights = []
            for _ in 0..<numFeedback  {
                outputGateFeedbackWeights.append(Gaussian.gaussianRandom(0.0, standardDeviation: 1.0 / Double(numFeedback)))    //  feedback weights - Initialize to a random number to break initial symmetry of the network, scaled to the inputs
            }
        }
    }
    
    func getNodeOutput(inputs: [Double], feedback: [Double], prevCellSum: Double) -> Double
    {
        //  Get the input gate value
        var inputGateSum = 0.0
        var sum = 0.0
        vDSP_dotprD(inputGateInputWeights, 1, inputs, 1, &inputGateSum, vDSP_Length(numInputs))
        vDSP_dotprD(inputGateFeedbackWeights, 1, feedback, 1, &sum, vDSP_Length(numFeedback))
        inputGateSum += sum
        let inputGateValue = 1.0 / (1.0 + exp(-inputGateSum))
        
        //  Get the forget gate value
        var forgetGateSum = 0.0
        vDSP_dotprD(forgetGateInputWeights, 1, inputs, 1, &forgetGateSum, vDSP_Length(numInputs))
        vDSP_dotprD(forgetGateFeedbackWeights, 1, feedback, 1, &sum, vDSP_Length(numFeedback))
        forgetGateSum += sum
        let forgetGateValue = 1.0 / (1.0 + exp(-forgetGateSum))
        
        //  Get the output gate value
        var outputGateSum = 0.0
        vDSP_dotprD(outputGateInputWeights, 1, inputs, 1, &outputGateSum, vDSP_Length(numInputs))
        vDSP_dotprD(outputGateFeedbackWeights, 1, feedback, 1, &sum, vDSP_Length(numFeedback))
        outputGateSum += sum
        let outputGateValue = 1.0 / (1.0 + exp(-outputGateSum))
        
        //  Get the memory cell z sumation
        var memoryCellSum = 0.0
        vDSP_dotprD(memoryCellInputWeights, 1, inputs, 1, &memoryCellSum, vDSP_Length(numInputs))
        vDSP_dotprD(memoryCellFeedbackWeights, 1, feedback, 1, &sum, vDSP_Length(numFeedback))
        memoryCellSum += sum
        
        //  Use the activation function function for the nonlinearity
        var memoryCellValue : Double
        switch (activation) {
        case .None:
            memoryCellValue = memoryCellSum
            break
        case .HyberbolicTangent:
            memoryCellValue = tanh(memoryCellSum)
            break
        case .SigmoidWithCrossEntropy:
            fallthrough
        case .Sigmoid:
            memoryCellValue = 1.0 / (1.0 + exp(-memoryCellSum))
            break
        case .RectifiedLinear:
            memoryCellValue = memoryCellSum
            if (memoryCellSum < 0) { memoryCellValue = 0.0 }
            break
        case .SoftSign:
            memoryCellValue = memoryCellSum / (1.0 + abs(memoryCellSum))
            break
        case .SoftMax:
            memoryCellValue = exp(memoryCellSum)
            break
        }
        
        //  Combine the forget and input gates into the cell summation
        let cellSum = prevCellSum * forgetGateValue + memoryCellValue * inputGateValue
        
        //  Use the activation function function for the nonlinearity
        var cellValue : Double
        switch (activation) {
        case .None:
            cellValue = cellSum
            break
        case .HyberbolicTangent:
            cellValue = tanh(cellSum)
            break
        case .SigmoidWithCrossEntropy:
            fallthrough
        case .Sigmoid:
            cellValue = 1.0 / (1.0 + exp(-cellSum))
            break
        case .RectifiedLinear:
            cellValue = cellSum
            if (cellSum < 0) { cellValue = 0.0 }
            break
        case .SoftSign:
            cellValue = cellSum / (1.0 + abs(cellSum))
            break
        case .SoftMax:
            cellValue = exp(cellSum)
            break
        }
        
        //  Multiply the cell value by the output gate value to get the final result
        lastOutput = cellValue * outputGateValue
        
        return lastOutput
    }
    
    //  Get the partial derivitive of the error with respect to the weighted sum
    func getFinalNodeDelta(expectedOutput: Double)
    {
        //  error = (result - expected value)^2  (squared error) - not the case for softmax or cross entropy
        //  derivitive of error = 2 * (result - expected value) * result'  (chain rule - result is a function of the sum through the non-linearity)
//        let 𝟃E𝟃h = 2.0 * (lastOutput - expectedOutput)      //  Squared error
        
        //  derivitive of the non-linearity: tanh' -> 1 - result², sigmoid -> result - result², rectlinear -> 0 if result<0 else 1
        //  derivitive of error = 2 * (result - expected value) * derivitive from above
        switch (activation) {
        case .None:
            delta = 2.0 * (lastOutput - expectedOutput)
            break
        case .HyberbolicTangent:
            delta = 2.0 * (lastOutput - expectedOutput) * (1 - lastOutput * lastOutput)
            break
        case .Sigmoid:
            delta = 2.0 * (lastOutput - expectedOutput) * (lastOutput - lastOutput * lastOutput)
            break
        case .SigmoidWithCrossEntropy:
            delta = (lastOutput - expectedOutput)
            break
        case .RectifiedLinear:
            delta = lastOutput < 0.0 ? 0.0 : 2.0 * (lastOutput - expectedOutput)
            break
        case .SoftSign:
            delta = (1-abs(lastOutput)) //  Start with derivitive for computation speed's sake
            delta *= delta
            delta *= 2.0 * (lastOutput - expectedOutput)
            break
        case .SoftMax:
            delta = (lastOutput - expectedOutput)
            break
        }
    }
}