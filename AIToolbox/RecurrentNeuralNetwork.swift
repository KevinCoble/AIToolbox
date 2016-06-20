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
    var W : [Double]        //  Weights for inputs from previous layer
    var U : [Double]        //  Weights for recurrent input data from this layer
    var h : Double //  Last result calculated
    var outputHistory : [Double] //  History of output for the sequence
    var 𝟃E𝟃h : Double      //  Gradient in error for this time step and future time steps with respect to output of this node
    var 𝟃E𝟃z : Double      //  Gradient of error with respect to weighted sum
    var 𝟃E𝟃W : [Double]   //  Accumulated weight W change gradient
    var 𝟃E𝟃U : [Double]   //  Accumulated weight U change gradient
    
    ///  Create the neural network node with a set activation function
    init(numInputs : Int, numFeedbacks : Int,  activationFunction: NeuralActivationFunction)
    {
        activation = activationFunction
        self.numInputs = numInputs + 1  //  Add one weight for the bias term
        self.numFeedback = numFeedbacks
        numWeights = self.numInputs + self.numFeedback
        W = []
        U = []
        h = 0.0
        outputHistory = []
        𝟃E𝟃h = 0.0
        𝟃E𝟃z = 0.0
        𝟃E𝟃W = []
        𝟃E𝟃U = []
    }
    
    //  Initialize the weights
    func initWeights(startWeights: [Double]!)
    {
        if let startWeights = startWeights {
            if (startWeights.count == 1) {
                W = [Double](count: numInputs, repeatedValue: startWeights[0])
                U = [Double](count: numFeedback, repeatedValue: startWeights[0])
            }
            else if (startWeights.count == numInputs+numFeedback) {
                //  Full weight array, just split into the two weight arrays
                W = Array(startWeights[0..<numInputs])
                U = Array(startWeights[numInputs..<numInputs+numFeedback])
            }
            else {
                W = []
                var index = 0 //  First number (if more than 1) goes into the bias weight, then repeat the initial
                for _ in 0..<numInputs-1  {
                    if (index >= startWeights.count-1) { index = 0 }      //  Wrap if necessary
                    W.append(startWeights[index])
                    index += 1
                }
                W.append(startWeights[startWeights.count-1])     //  Add the bias term
                
                index = 0
                U = []
                for _ in 0..<numFeedback  {
                    if (index >= startWeights.count-1) { index = 1 }      //  Wrap if necessary
                    U.append(startWeights[index])
                    index += 1
                }
            }
        }
        else {
            W = []
            for _ in 0..<numInputs-1  {
                W.append(Gaussian.gaussianRandom(0.0, standardDeviation: 1.0 / Double(numInputs-1)))    //  input weights - Initialize to a random number to break initial symmetry of the network, scaled to the inputs
            }
            W.append(Gaussian.gaussianRandom(0.0, standardDeviation:1.0))    //  Bias weight - Initialize to a  random number to break initial symmetry of the network
            
            U = []
            for _ in 0..<numFeedback  {
                U.append(Gaussian.gaussianRandom(0.0, standardDeviation: 1.0 / Double(numFeedback)))    //  feedback weights - Initialize to a random number to break initial symmetry of the network, scaled to the inputs
            }
        }
    }
    
    func feedForward(x: [Double], hPrev: [Double]) -> Double
    {
        //  Get the weighted sum:  z = W⋅x + U⋅h(t-1)
        var z = 0.0
        var sum = 0.0
        vDSP_dotprD(W, 1, x, 1, &z, vDSP_Length(numInputs))
        vDSP_dotprD(U, 1, hPrev, 1, &sum, vDSP_Length(numFeedback))
        z += sum
        
        //  Use the activation function function for the nonlinearity:  h = act(z)
        switch (activation) {
        case .None:
            h = z
            break
        case .HyperbolicTangent:
            h = tanh(z)
            break
        case .SigmoidWithCrossEntropy:
            fallthrough
        case .Sigmoid:
            h = 1.0 / (1.0 + exp(-z))
            break
        case .RectifiedLinear:
            h = z
            if (z < 0) { h = 0.0 }
            break
        case .SoftSign:
            h = z / (1.0 + abs(z))
            break
        case .SoftMax:
            h = exp(z)
            break
        }
        
        return h
    }
    
    //  Get the partial derivitive of the error with respect to the weighted sum
    func getFinalNode𝟃E𝟃zs(𝟃E𝟃h: Double)
    {
        //  Calculate 𝟃E/𝟃z.  𝟃E/𝟃z = 𝟃E/𝟃h ⋅ 𝟃h/𝟃z = 𝟃E/𝟃h ⋅ derivitive of nonlinearity
        //  derivitive of the non-linearity: tanh' -> 1 - result^2, sigmoid -> result - result^2, rectlinear -> 0 if result<0 else 1
        switch (activation) {
        case .None:
            𝟃E𝟃z = 𝟃E𝟃h
            break
        case .HyperbolicTangent:
            𝟃E𝟃z = 𝟃E𝟃h * (1 - h * h)
            break
        case .Sigmoid:
            𝟃E𝟃z = 𝟃E𝟃h * (h - h * h)
            break
        case .SigmoidWithCrossEntropy:
            𝟃E𝟃z = 𝟃E𝟃h
            break
        case .RectifiedLinear:
            𝟃E𝟃z = h <= 0.0 ? 0.0 : 𝟃E𝟃h
            break
        case .SoftSign:
            //  Reconstitute z from h
            var z : Double
            if (h < 0) {        //  Negative z
                z = h / (1.0 + h)
                𝟃E𝟃z = -𝟃E𝟃h / ((1.0 + z) * (1.0 + z))
            }
            else {              //  Positive z
                z = h / (1.0 - h)
                𝟃E𝟃z = 𝟃E𝟃h / ((1.0 + z) * (1.0 + z))
            }
            break
        case .SoftMax:
            𝟃E𝟃z = 𝟃E𝟃h
            break
        }
    }
    
    func reset𝟃E𝟃hs()
    {
        𝟃E𝟃h = 0.0
    }
    
    func addTo𝟃E𝟃hs(addition: Double)
    {
        𝟃E𝟃h += addition
    }
    
    func getWeightTimes𝟃E𝟃zs(weightIndex: Int) ->Double
    {
        return W[weightIndex] * 𝟃E𝟃z
    }
    
    func getFeedbackWeightTimes𝟃E𝟃zs(weightIndex: Int) ->Double
    {
        return U[weightIndex] * 𝟃E𝟃z
    }
    
    func get𝟃E𝟃z()
    {
        //  𝟃E𝟃h contains 𝟃E/𝟃h for the current time step plus all future time steps.
        
        //  Calculate 𝟃E𝟃z.   𝟃E/𝟃z = 𝟃E/𝟃h ⋅ 𝟃h/𝟃z  =  𝟃E/𝟃h ⋅ derivitive of non-linearity
        //  derivitive of the non-linearity: tanh' -> 1 - result^2, sigmoid -> result - result^2, rectlinear -> 0 if result<0 else 1
        switch (activation) {
        case .None:
            break
        case .HyperbolicTangent:
            𝟃E𝟃z = 𝟃E𝟃h * (1 - h * h)
            break
        case .SigmoidWithCrossEntropy:
            fallthrough
        case .Sigmoid:
            𝟃E𝟃z = 𝟃E𝟃h * (h - h * h)
            break
        case .RectifiedLinear:
            𝟃E𝟃z = h < 0.0 ? 0.0 : 𝟃E𝟃h
            break
        case .SoftSign:
            //  Reconstitute z from h
            var z : Double
            if (h < 0) {        //  Negative z
                z = h / (1.0 + h)
                𝟃E𝟃z = -𝟃E𝟃h / ((1.0 + z) * (1.0 + z))
            }
            else {              //  Positive z
                z = h / (1.0 - h)
                𝟃E𝟃z = 𝟃E𝟃h / ((1.0 + z) * (1.0 + z))
            }
            break
        case .SoftMax:
            //  Should not get here - SoftMax is only valid on output layer
            break
        }
    }
    
    func clearWeightChanges()
    {
        𝟃E𝟃W = [Double](count: numInputs, repeatedValue: 0.0)
        𝟃E𝟃U = [Double](count: numFeedback, repeatedValue: 0.0)
    }
    
    func appendWeightChanges(x: [Double], hPrev: [Double]) -> Double
    {
        //  Update each weight accumulation
        //  z = W⋅x + U⋅hPrev, therefore
        //      𝟃E/𝟃W = 𝟃E/𝟃z ⋅ 𝟃z/𝟃W = 𝟃E/𝟃z ⋅  x
        //      𝟃E/𝟃U = 𝟃E/𝟃z ⋅ 𝟃z/𝟃U = 𝟃E/𝟃z ⋅  hPrev
        
        //  𝟃E/𝟃W += 𝟃E/𝟃z ⋅ 𝟃z/𝟃W = 𝟃E/𝟃z ⋅ x
        vDSP_vsmaD(x, 1, &𝟃E𝟃z, 𝟃E𝟃W, 1, &𝟃E𝟃W, 1, vDSP_Length(numInputs))
        
        //  𝟃E/𝟃U += 𝟃E/𝟃z ⋅ 𝟃z/𝟃U = 𝟃E/𝟃z ⋅ hPrev
        vDSP_vsmaD(hPrev, 1, &𝟃E𝟃z, 𝟃E𝟃U, 1, &𝟃E𝟃U, 1, vDSP_Length(numFeedback))
        
        return h     //  return output for next layer
    }
    
    func updateWeightsFromAccumulations(averageTrainingRate: Double)
    {
        //  Update the weights from the accumulations
        //  weights -= accumulation * averageTrainingRate
        var η = -averageTrainingRate     //  Needed for unsafe pointer conversion  - negate for multiply-and-add vector operation
        vDSP_vsmaD(𝟃E𝟃W, 1, &η, W, 1, &W, 1, vDSP_Length(numInputs))
        vDSP_vsmaD(𝟃E𝟃U, 1, &η, U, 1, &U, 1, vDSP_Length(numFeedback))
    }
    
    func decayWeights(decayFactor : Double)
    {
        var λ = decayFactor     //  Needed for unsafe pointer conversion
        vDSP_vsmulD(W, 1, &λ, &W, 1, vDSP_Length(numInputs-1))
        vDSP_vsmulD(U, 1, &λ, &U, 1, vDSP_Length(numFeedback))
    }
    
    func resetSequence()
    {
        h = 0.0
        outputHistory = [0.0]       //  first 'previous' value is zero
        𝟃E𝟃z = 0.0                 //  Backward propogation previous 𝟃E𝟃z (𝟃E𝟃z from next time step in sequence) is zero
    }
    
    func storeRecurrentValues()
    {
        outputHistory.append(h)
    }
    
    func getLastRecurrentValue()
    {
        h = outputHistory.removeLast()
    }
    
    func getPreviousOutputValue() -> Double
    {
        let hPrev = outputHistory.last
        if (hPrev == nil) { return 0.0 }
        return hPrev!
    }
}

final class RecurrentNeuralLayer: NeuralLayer {
    //  Nodes
    var nodes : [RecurrentNeuralNode]
    var bpttSequenceIndex: Int
    
    ///  Create the neural network layer based on a tuple (number of nodes, activation function)
    init(numInputs : Int, layerDefinition: (layerType: NeuronLayerType, numNodes: Int, activation: NeuralActivationFunction, auxiliaryData: AnyObject?))
    {
        nodes = []
        for _ in 0..<layerDefinition.numNodes {
            nodes.append(RecurrentNeuralNode(numInputs: numInputs, numFeedbacks: layerDefinition.numNodes, activationFunction: layerDefinition.activation))
        }
        bpttSequenceIndex = 0
    }
    
    //  Initialize the weights
    func initWeights(startWeights: [Double]!)
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
            weights += node.W
            weights += node.U
        }
        return weights
    }
    
    func getLastOutput() -> [Double]
    {
        var h: [Double] = []
        for node in nodes {
            h.append(node.h)
        }
        return h
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
    
    func feedForward(x: [Double]) -> [Double]
    {
        //  Gather the previous outputs for the feedback
        var hPrev : [Double] = []
        for node in nodes {
            hPrev.append(node.h)
        }
        
        var outputs : [Double] = []
        //  Assume input array already has bias constant 1.0 appended
        //  Fully-connected nodes means all nodes get the same input array
        if (nodes[0].activation == .SoftMax) {
            var sum = 0.0
            for node in nodes {     //  Sum each output
                sum += node.feedForward(x, hPrev: hPrev)
            }
            let scale = 1.0 / sum       //  Do division once for efficiency
            for node in nodes {     //  Get the outputs scaled by the sum to give the probability distribuition for the output
                node.h *= scale
                outputs.append(node.h)
            }
        }
        else {
            for node in nodes {
                outputs.append(node.feedForward(x, hPrev: hPrev))
            }
        }
        
        return outputs
    }
    
    func getFinalLayer𝟃E𝟃zs(𝟃E𝟃h: [Double])
    {
        for nNodeIndex in 0..<nodes.count {
            //  Start with the portion from the squared error term
            nodes[nNodeIndex].getFinalNode𝟃E𝟃zs(𝟃E𝟃h[nNodeIndex])
        }
    }
    
    func getLayer𝟃E𝟃zs(nextLayer: NeuralLayer)
    {
        //  Get 𝟃E/𝟃h
        for nNodeIndex in 0..<nodes.count {
            nodes[nNodeIndex].reset𝟃E𝟃hs()
            
            //  Add each portion from the nodes in the next forward layer to get 𝟃Enow/𝟃h
            nodes[nNodeIndex].addTo𝟃E𝟃hs(nextLayer.get𝟃E𝟃hForNodeInPreviousLayer(nNodeIndex))
            
            //  Add each portion from the nodes in this layer, using the feedback weights.  This adds 𝟃Efuture/𝟃h
            for node in nodes {
                nodes[nNodeIndex].addTo𝟃E𝟃hs(node.getFeedbackWeightTimes𝟃E𝟃zs(nNodeIndex))
            }
        }
        
        //  Calculate 𝟃E/𝟃z from 𝟃E/𝟃h
        for node in nodes {
            node.get𝟃E𝟃z()
        }
    }
    
    func get𝟃E𝟃hForNodeInPreviousLayer(inputIndex: Int) ->Double
    {
        var sum = 0.0
        for node in nodes {
            sum += node.getWeightTimes𝟃E𝟃zs(inputIndex)
        }
        return sum
    }
    
    func clearWeightChanges()
    {
        for node in nodes {
            node.clearWeightChanges()
        }
    }
    
    func appendWeightChanges(x: [Double]) -> [Double]
    {
        //  Gather the previous outputs for the feedback
        var hPrev : [Double] = []
        for node in nodes {
            hPrev.append(node.getPreviousOutputValue())
        }
        
        var outputs : [Double] = []
        //  Assume input array already has bias constant 1.0 appended
        //  Fully-connected nodes means all nodes get the same input array
        for node in nodes {
            outputs.append(node.appendWeightChanges(x, hPrev: hPrev))
        }
        
        return outputs
    }
    
    func updateWeightsFromAccumulations(averageTrainingRate: Double, weightDecay: Double)
    {
        //  Have each node update it's weights from the accumulations
        for node in nodes {
            if (weightDecay < 1) { node.decayWeights(weightDecay) }
            node.updateWeightsFromAccumulations(averageTrainingRate)
        }
    }
    
    func decayWeights(decayFactor : Double)
    {
        for node in nodes {
            node.decayWeights(decayFactor)
        }
    }
    
    func getSingleNodeClassifyValue() -> Double
    {
        let activation = nodes[0].activation
        if (activation == .HyperbolicTangent || activation == .RectifiedLinear) { return 0.0 }
        return 0.5
    }
    
    func resetSequence()
    {
        //  Have each node reset
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
    
    func retrieveRecurrentValues(sequenceIndex: Int)
    {
        bpttSequenceIndex =  sequenceIndex
        
        //  Set the last recurrent value in the history array to the last output
        for node in nodes {
            node.getLastRecurrentValue()
        }
    }
}
