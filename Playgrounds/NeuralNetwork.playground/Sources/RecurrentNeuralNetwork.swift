//
//  RecurrentNeuralNetwork.swift
//  AIToolbox
//
//  Created by Kevin Coble on 5/5/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation
import Accelerate


final class RecurrentNeuralNode {
    //  Activation function
    let activation : NeuralActivationFunction
    let numWeights : Int        //  This includes weights from inputs and from feedback
    let numInputs : Int
    let numFeedback : Int
    var inputWeights : [Double]
    var feedbackWeights : [Double]
    var lastWeightedSum : Double //  Last weights dot-producted with inputs - remembered for training purposes
    var lastOutput : Double //  Last result calculated
    var outputHistory : [Double] //  History of output for the sequence
    var delta : Double      //  Difference in expected output to calculated result - weighted sum from all nodes this node outputs too
    var forwardDelta : Double
    var accumulatedInputWeightChanges : [Double]?
    var accumulatedFeedbackWeightChanges : [Double]?
    
    ///  Create the neural network node with a set activation function
    init(numInputs : Int, numFeedbacks : Int,  activationFunction: NeuralActivationFunction)
    {
        activation = activationFunction
        self.numInputs = numInputs + 1  //  Add one weight for the bias term
        self.numFeedback = numFeedbacks
        numWeights = self.numInputs + self.numFeedback
        inputWeights = []
        feedbackWeights = []
        lastWeightedSum = 0.0
        lastOutput = 0.0
        outputHistory = []
        delta = 0.0
        forwardDelta = 0.0
    }
    
    //  Initialize the weights
    func initWeights(_ startWeights: [Double]!)
    {
        if let startWeights = startWeights {
            if (startWeights.count == 1) {
                inputWeights = [Double](repeating: startWeights[0], count: numInputs)
                feedbackWeights = [Double](repeating: startWeights[0], count: numFeedback)
            }
            else if (startWeights.count == numInputs+numFeedback) {
                //  Full weight array, just split into the two weight arrays
                inputWeights = Array(startWeights[0..<numInputs])
                feedbackWeights = Array(startWeights[numInputs..<numInputs+numFeedback])
            }
            else {
                inputWeights = []
                var index = 0 //  First number (if more than 1) goes into the bias weight, then repeat the initial
                for _ in 0..<numInputs-1  {
                    if (index >= startWeights.count-1) { index = 0 }      //  Wrap if necessary
                    inputWeights.append(startWeights[index])
                    index += 1
                }
                inputWeights.append(startWeights[startWeights.count-1])     //  Add the bias term
                
                index = 0
                feedbackWeights = []
                for _ in 0..<numFeedback  {
                    if (index >= startWeights.count-1) { index = 1 }      //  Wrap if necessary
                    feedbackWeights.append(startWeights[index])
                    index += 1
                }
            }
        }
        else {
            inputWeights = []
            for _ in 0..<numInputs-1  {
                inputWeights.append(Gaussian.gaussianRandom(0.0, standardDeviation: 1.0 / Double(numInputs-1)))    //  input weights - Initialize to a random number to break initial symmetry of the network, scaled to the inputs
            }
            inputWeights.append(Gaussian.gaussianRandom(0.0, standardDeviation:1.0))    //  Bias weight - Initialize to a  random number to break initial symmetry of the network
            
            feedbackWeights = []
            for _ in 0..<numFeedback  {
                feedbackWeights.append(Gaussian.gaussianRandom(0.0, standardDeviation: 1.0 / Double(numFeedback)))    //  feedback weights - Initialize to a random number to break initial symmetry of the network, scaled to the inputs
            }
        }
    }
    
    func getNodeOutput(_ inputs: [Double], feedback: [Double]) -> Double
    {
        //  Get the weighted sum
        var sum = 0.0
        vDSP_dotprD(inputWeights, 1, inputs, 1, &lastWeightedSum, vDSP_Length(numInputs))
        vDSP_dotprD(feedbackWeights, 1, feedback, 1, &sum, vDSP_Length(numFeedback))
        lastWeightedSum += sum
        
        //  Use the activation function function for the nonlinearity
        switch (activation) {
        case .none:
            lastOutput = lastWeightedSum
            break
        case .hyberbolicTangent:
            lastOutput = tanh(lastWeightedSum)
            break
        case .sigmoidWithCrossEntropy:
            fallthrough
        case .sigmoid:
            lastOutput = 1.0 / (1.0 + exp(-lastWeightedSum))
            break
        case .rectifiedLinear:
            lastOutput = lastWeightedSum
            if (lastWeightedSum < 0) { lastOutput = 0.0 }
            break
        case .softSign:
            lastOutput = lastWeightedSum / (1.0 + abs(lastWeightedSum))
            break
        case .softMax:
            lastOutput = exp(lastWeightedSum)
            break
        }
        
        return lastOutput
    }
    
    //  Get the partial derivitive of the error with respect to the weighted sum
    func getFinalNodeDelta(_ expectedOutput: Double)
    {
        //  error = (result - expected value)^2  (squared error) - not the case for softmax or cross entropy
        //  derivitive of error = 2 * (result - expected value) * result'  (chain rule - result is a function of the sum through the non-linearity)
        //  derivitive of the non-linearity: tanh' -> 1 - result², sigmoid -> result - result², rectlinear -> 0 if result<0 else 1
        //  derivitive of error = 2 * (result - expected value) * derivitive from above
        switch (activation) {
        case .none:
            delta = 2.0 * (lastOutput - expectedOutput)
            break
        case .hyberbolicTangent:
            delta = 2.0 * (lastOutput - expectedOutput) * (1 - lastOutput * lastOutput)
            break
        case .sigmoid:
            delta = 2.0 * (lastOutput - expectedOutput) * (lastOutput - lastOutput * lastOutput)
            break
        case .sigmoidWithCrossEntropy:
            delta = (lastOutput - expectedOutput)
            break
        case .rectifiedLinear:
            delta = lastOutput < 0.0 ? 0.0 : 2.0 * (lastOutput - expectedOutput)
            break
        case .softSign:
            delta = (1-abs(lastOutput)) //  Start with derivitive for computation speed's sake
            delta *= delta
            delta *= 2.0 * (lastOutput - expectedOutput)
            break
        case .softMax:
            delta = (lastOutput - expectedOutput)
            break
        }
    }
    
    func resetDelta()
    {
        forwardDelta = delta
        delta = 0.0
    }
    
    func addToDelta(_ addition: Double)
    {
        delta += addition
    }
    
    func getWeightTimesDelta(_ weightIndex: Int) ->Double
    {
        return inputWeights[weightIndex] * delta
    }
    
    func getRecurrentWeightTimesForwardDelta(_ weightIndex: Int) ->Double
    {
        return feedbackWeights[weightIndex] * forwardDelta
    }
    
    func multiplyDeltaByNonLinearityDerivitive()
    {
        //  derivitive of the non-linearity: tanh' -> 1 - result^2, sigmoid -> result - result^2, rectlinear -> 0 if result<0 else 1
        switch (activation) {
        case .none:
            break
        case .hyberbolicTangent:
            delta *= (1 - lastOutput * lastOutput)
            break
        case .sigmoidWithCrossEntropy:
            fallthrough
        case .sigmoid:
            delta *= (lastOutput - lastOutput * lastOutput)
            break
        case .rectifiedLinear:
            delta = lastOutput < 0.0 ? 0.0 : delta
            break
        case .softSign:
            if (lastOutput < 0) { delta *= -1 }
            delta /= (1.0 + lastOutput) * (1.0 + lastOutput)
            break
        case .softMax:
            //  Should not get here - SoftMax is only valid on output layer
            break
        }
    }
    
    func updateWeights(_ inputs: [Double], feedback: [Double], trainingRate: Double) -> Double
    {
        //  Update each weight
        var negativeScaleFactor = trainingRate * delta * -1.0  //  Make negative so we can use DSP to vectorize equation
        vDSP_vsmaD(inputs, 1, &negativeScaleFactor, inputWeights, 1, &inputWeights, 1, vDSP_Length(numInputs))    //  weights = weights + delta * inputs
        vDSP_vsmaD(feedback, 1, &negativeScaleFactor, feedbackWeights, 1, &feedbackWeights, 1, vDSP_Length(numFeedback))    //  weights = weights + delta * feedback inputs
        
        return lastOutput
    }
    
    func clearWeightChanges()
    {
        accumulatedInputWeightChanges = [Double](repeating: 0.0, count: numInputs)
        accumulatedFeedbackWeightChanges = [Double](repeating: 0.0, count: numFeedback)
    }
    
    func appendWeightChanges(_ inputs: [Double], feedback: [Double]) -> Double
    {
        //  Update each weight accumulation
        vDSP_vsmaD(inputs, 1, &delta, accumulatedInputWeightChanges!, 1, &accumulatedInputWeightChanges!, 1, vDSP_Length(numInputs))       //  Calculate the weight change:  total change = total change + delta * inputs
        vDSP_vsmaD(feedback, 1, &delta, accumulatedFeedbackWeightChanges!, 1, &accumulatedFeedbackWeightChanges!, 1, vDSP_Length(numFeedback))       //  Calculate the weight change:  total change = total change + delta * feedback
        
        return lastOutput
    }
    
    func updateWeightsFromAccumulations(_ averageTrainingRate: Double)
    {
        //  Update the weights from the accumulations
        //  weights -= accumulation * averageTrainingRate  (training rate is passed in negative to allow subtraction with vector math)
        var x = averageTrainingRate     //  Needed for unsafe pointer conversion
        vDSP_vsmaD(accumulatedInputWeightChanges!, 1, &x, inputWeights, 1, &inputWeights, 1, vDSP_Length(numInputs))
        vDSP_vsmaD(accumulatedFeedbackWeightChanges!, 1, &x, feedbackWeights, 1, &feedbackWeights, 1, vDSP_Length(numFeedback))
    }
    
    func decayWeights(_ decayFactor : Double)
    {
        var x = decayFactor     //  Needed for unsafe pointer conversion
        vDSP_vsmulD(inputWeights, 1, &x, &inputWeights, 1, vDSP_Length(numInputs-1))
        vDSP_vsmulD(feedbackWeights, 1, &x, &feedbackWeights, 1, vDSP_Length(numFeedback))
    }
    
    func resetSequence()
    {
        lastOutput = 0.0
        outputHistory = [0.0]       //  first 'previous' value is zero
        delta = 0.0                 //  Backward propogation previous delta (delta from next time step in sequence) is zero
    }
    
    func storeRecurrentValues()
    {
        outputHistory.append(lastOutput)
    }
    
    func getLastRecurrentValue()
    {
        lastOutput = outputHistory.removeLast()
    }
    
    func getPreviousOutputValue() -> Double
    {
        let prevValue = outputHistory.last
        if (prevValue == nil) { return 0.0 }
        return prevValue!
    }

}

final class RecurrentNeuralLayer: NeuralLayer {
    //  Nodes
    var nodes : [RecurrentNeuralNode]
    
    ///  Create the neural network layer based on a tuple (number of nodes, activation function)
    init(numInputs : Int, layerDefinition: (layerType: NeuronLayerType, numNodes: Int, activation: NeuralActivationFunction, auxiliaryData: AnyObject?))
    {
        nodes = []
        for _ in 0..<layerDefinition.numNodes {
            nodes.append(RecurrentNeuralNode(numInputs: numInputs, numFeedbacks: layerDefinition.numNodes, activationFunction: layerDefinition.activation))
        }
    }
    
    //  Initialize the weights
    func initWeights(_ startWeights: [Double]!)
    {
        if let startWeights = startWeights {
            if (startWeights.count >= nodes.count * nodes[0].numWeights) {
                //  If there are enough weights for all nodes, split the weights and initialize
                var startIndex = 0
                for node in nodes {
                    let subArray = Array(startWeights[startIndex...(startIndex+node.numWeights-1)])
                    node.initWeights(subArray)
                    startIndex += node.numWeights
                }
            }
            else {
                //  If there are not enough weights for all nodes, initialize each node with the set given
                for node in nodes {
                    node.initWeights(startWeights)
                }
            }
        }
        else {
            //  No specified weights - just initialize normally
            for node in nodes {
                node.initWeights(nil)
            }
        }
    }
    
    func getWeights() -> [Double]
    {
        var weights: [Double] = []
        for node in nodes {
            weights += node.inputWeights
            weights += node.feedbackWeights
        }
        return weights
    }
    
    func getNodeCount() -> Int
    {
        return nodes.count
    }
    
    func getWeightsPerNode()-> Int
    {
        return nodes[0].numWeights
    }
    
    func getActivation()-> NeuralActivationFunction
    {
        return nodes[0].activation
    }
    
    func getLayerOutputs(_ inputs: [Double]) -> [Double]
    {
        //  Gather the previous outputs for the feedback
        var feedback : [Double] = []
        for node in nodes {
            feedback.append(node.lastOutput)
        }
        
        var outputs : [Double] = []
        //  Assume input array already has bias constant 1.0 appended
        //  Fully-connected nodes means all nodes get the same input array
        if (nodes[0].activation == .softMax) {
            var sum = 0.0
            for node in nodes {     //  Sum each output
                sum += node.getNodeOutput(inputs, feedback: feedback)
            }
            let scale = 1.0 / sum       //  Do division once for efficiency
            for node in nodes {     //  Get the outputs scaled by the sum to give the probability distribuition for the output
                node.lastOutput *= scale
                outputs.append(node.lastOutput)
            }
        }
        else {
            for node in nodes {
                outputs.append(node.getNodeOutput(inputs, feedback: feedback))
            }
        }
        
        return outputs
    }
    
    func getFinalLayerDelta(_ expectedOutputs: [Double])
    {
        for nodeIndex in 0..<nodes.count {
            nodes[nodeIndex].getFinalNodeDelta(expectedOutputs[nodeIndex])
        }
    }
    
    func getLayerDelta(_ nextLayer: NeuralLayer)
    {
        for nNodeIndex in 0..<nodes.count {
            nodes[nNodeIndex].resetDelta()
            
            //  Add each portion from the nodes in the next forward layer
            nodes[nNodeIndex].addToDelta(nextLayer.getSumOfWeightsTimesDelta(nNodeIndex))
            
            //  Add each portion from the feedback
            for node in nodes {
                nodes[nNodeIndex].addToDelta(node.getRecurrentWeightTimesForwardDelta(nNodeIndex))
            }
            
            //  Multiply by the non-linearity derivitive
            nodes[nNodeIndex].multiplyDeltaByNonLinearityDerivitive()
        }
    }
    
    func getSumOfWeightsTimesDelta(_ weightIndex: Int) ->Double
    {
        var sum = 0.0
        for node in nodes {
            sum += node.getWeightTimesDelta(weightIndex)
        }
        return sum
    }
    
    func updateWeights(_ inputs: [Double], trainingRate: Double, weightDecay: Double) -> [Double]
    {
        //  Gather the previous outputs for the feedback
        var feedback : [Double] = []
        for node in nodes {
            feedback.append(node.getPreviousOutputValue())
        }
        
        var outputs : [Double] = []
        //  Assume input array already has bias constant 1.0 appended
        //  Fully-connected nodes means all nodes get the same input array
        for node in nodes {
            if (weightDecay < 1) { node.decayWeights(weightDecay) }
            outputs.append(node.updateWeights(inputs, feedback: feedback, trainingRate: trainingRate))
        }
        
        return outputs
    }
    
    func clearWeightChanges()
    {
        for node in nodes {
            node.clearWeightChanges()
        }
    }
    
    func appendWeightChanges(_ inputs: [Double]) -> [Double]
    {
        //  Gather the previous outputs for the feedback
        var feedback : [Double] = []
        for node in nodes {
            feedback.append(node.getPreviousOutputValue())
        }
        
        var outputs : [Double] = []
        //  Assume input array already has bias constant 1.0 appended
        //  Fully-connected nodes means all nodes get the same input array
        for node in nodes {
            outputs.append(node.appendWeightChanges(inputs, feedback: feedback))
        }
        
        return outputs
    }
    
    func updateWeightsFromAccumulations(_ averageTrainingRate: Double, weightDecay: Double)
    {
        //  Have each node update it's weights from the accumulations
        for node in nodes {
            if (weightDecay < 1) { node.decayWeights(weightDecay) }
            node.updateWeightsFromAccumulations(averageTrainingRate)
        }
    }
    
    func decayWeights(_ decayFactor : Double)
    {
        for node in nodes {
            node.decayWeights(decayFactor)
        }
    }
    
    func getSingleNodeClassifyValue() -> Double
    {
        let activation = nodes[0].activation
        if (activation == .hyberbolicTangent || activation == .rectifiedLinear) { return 0.0 }
        return 0.5
    }
    
    func resetSequence()
    {
        for node in nodes {
            node.resetSequence()
        }
    }
    
    func storeRecurrentValues()
    {
        for node in nodes {
            node.storeRecurrentValues()
        }
    }
    
    func retrieveRecurrentValues()
    {
        //  Set the last recurrent value in the history array to the last output
        for node in nodes {
            node.getLastRecurrentValue()
        }
    }
}
