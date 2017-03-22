//
//  RecurrentNeuralNetwork.swift
//  AIToolbox
//
//  Created by Kevin Coble on 5/5/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation
#if os(Linux)
#else
import Accelerate
#endif


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
    var weightUpdateMethod = NeuralWeightUpdateMethod.normal
    var weightUpdateParameter : Double?      //  Decay rate for rms prop weight updates
    var WweightUpdateData : [Double] = []    //  Array of running average for rmsprop
    var UweightUpdateData : [Double] = []    //  Array of running average for rmsprop
    
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
    func initWeights(_ startWeights: [Double]!)
    {
        if let startWeights = startWeights {
            if (startWeights.count == 1) {
                W = [Double](repeating: startWeights[0], count: numInputs)
                U = [Double](repeating: startWeights[0], count: numFeedback)
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
        
        //  If rmsprop update, allocate the momentum storage array
        if (weightUpdateMethod == .rmsProp) {
            WweightUpdateData = [Double](repeating: 0.0, count: numInputs)
            UweightUpdateData = [Double](repeating: 0.0, count: numFeedback)
        }
    }
    
    func setNeuralWeightUpdateMethod(_ method: NeuralWeightUpdateMethod, _ parameter: Double?)
    {
        weightUpdateMethod = method
        weightUpdateParameter = parameter
    }
    
    func feedForward(_ x: [Double], hPrev: [Double]) -> Double
    {
        //  Get the weighted sum:  z = W⋅x + U⋅h(t-1)
        var z = 0.0
        var sum = 0.0
        vDSP_dotprD(W, 1, x, 1, &z, vDSP_Length(numInputs))
        vDSP_dotprD(U, 1, hPrev, 1, &sum, vDSP_Length(numFeedback))
        z += sum
        
        //  Use the activation function function for the nonlinearity:  h = act(z)
        switch (activation) {
        case .none:
            h = z
            break
        case .hyperbolicTangent:
            h = tanh(z)
            break
        case .sigmoidWithCrossEntropy:
            fallthrough
        case .sigmoid:
            h = 1.0 / (1.0 + exp(-z))
            break
        case .rectifiedLinear:
            h = z
            if (z < 0) { h = 0.0 }
            break
        case .softSign:
            h = z / (1.0 + abs(z))
            break
        case .softMax:
            h = exp(z)
            break
        }
        
        return h
    }
    
    //  Get the partial derivitive of the error with respect to the weighted sum
    func getFinalNode𝟃E𝟃zs(_ 𝟃E𝟃h: Double)
    {
        //  Calculate 𝟃E/𝟃z.  𝟃E/𝟃z = 𝟃E/𝟃h ⋅ 𝟃h/𝟃z = 𝟃E/𝟃h ⋅ derivitive of nonlinearity
        //  derivitive of the non-linearity: tanh' -> 1 - result^2, sigmoid -> result - result^2, rectlinear -> 0 if result<0 else 1
        switch (activation) {
        case .none:
            𝟃E𝟃z = 𝟃E𝟃h
            break
        case .hyperbolicTangent:
            𝟃E𝟃z = 𝟃E𝟃h * (1 - h * h)
            break
        case .sigmoid:
            𝟃E𝟃z = 𝟃E𝟃h * (h - h * h)
            break
        case .sigmoidWithCrossEntropy:
            𝟃E𝟃z = 𝟃E𝟃h
            break
        case .rectifiedLinear:
            𝟃E𝟃z = h <= 0.0 ? 0.0 : 𝟃E𝟃h
            break
        case .softSign:
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
        case .softMax:
            𝟃E𝟃z = 𝟃E𝟃h
            break
        }
    }
    
    func reset𝟃E𝟃hs()
    {
        𝟃E𝟃h = 0.0
    }
    
    func addTo𝟃E𝟃hs(_ addition: Double)
    {
        𝟃E𝟃h += addition
    }
    
    func getWeightTimes𝟃E𝟃zs(_ weightIndex: Int) ->Double
    {
        return W[weightIndex] * 𝟃E𝟃z
    }
    
    func getFeedbackWeightTimes𝟃E𝟃zs(_ weightIndex: Int) ->Double
    {
        return U[weightIndex] * 𝟃E𝟃z
    }
    
    func get𝟃E𝟃z()
    {
        //  𝟃E𝟃h contains 𝟃E/𝟃h for the current time step plus all future time steps.
        
        //  Calculate 𝟃E𝟃z.   𝟃E/𝟃z = 𝟃E/𝟃h ⋅ 𝟃h/𝟃z  =  𝟃E/𝟃h ⋅ derivitive of non-linearity
        //  derivitive of the non-linearity: tanh' -> 1 - result^2, sigmoid -> result - result^2, rectlinear -> 0 if result<0 else 1
        switch (activation) {
        case .none:
            break
        case .hyperbolicTangent:
            𝟃E𝟃z = 𝟃E𝟃h * (1 - h * h)
            break
        case .sigmoidWithCrossEntropy:
            fallthrough
        case .sigmoid:
            𝟃E𝟃z = 𝟃E𝟃h * (h - h * h)
            break
        case .rectifiedLinear:
            𝟃E𝟃z = h < 0.0 ? 0.0 : 𝟃E𝟃h
            break
        case .softSign:
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
        case .softMax:
            //  Should not get here - SoftMax is only valid on output layer
            break
        }
    }
    
    func clearWeightChanges()
    {
        𝟃E𝟃W = [Double](repeating: 0.0, count: numInputs)
        𝟃E𝟃U = [Double](repeating: 0.0, count: numFeedback)
    }
    
    func appendWeightChanges(_ x: [Double], hPrev: [Double]) -> Double
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
    
    func updateWeightsFromAccumulations(_ averageTrainingRate: Double)
    {
        //  Update the weights from the accumulations
        switch weightUpdateMethod {
        case .normal:
            //  W -= 𝟃E/𝟃W * averageTrainingRate, U -= 𝟃E/𝟃U * averageTrainingRate
            var η = -averageTrainingRate     //  Needed for unsafe pointer conversion - negate for multiply-and-add vector operation
            vDSP_vsmaD(𝟃E𝟃W, 1, &η, W, 1, &W, 1, vDSP_Length(numInputs))
            vDSP_vsmaD(𝟃E𝟃U, 1, &η, U, 1, &U, 1, vDSP_Length(numFeedback))
        case .rmsProp:
            //  Update the rmsProp cache for W --> rmsprop_cache = decay_rate * rmsprop_cache + (1 - decay_rate) * gradient²
            var gradSquared = [Double](repeating: 0.0, count: numInputs)
            vDSP_vsqD(𝟃E𝟃W, 1, &gradSquared, 1, vDSP_Length(numInputs))  //  Get the gradient squared
            var decay = 1.0 - weightUpdateParameter!
            vDSP_vsmulD(gradSquared, 1, &decay, &gradSquared, 1, vDSP_Length(numInputs))   //  (1 - decay_rate) * gradient²
            decay = weightUpdateParameter!
            vDSP_vsmaD(WweightUpdateData, 1, &decay, gradSquared, 1, &WweightUpdateData, 1, vDSP_Length(numInputs))
            //  Update the weights --> weight += learning_rate * gradient / (sqrt(rmsprop_cache) + 1e-5)
            for i in 0..<numInputs { gradSquared[i] = sqrt(WweightUpdateData[i]) }      //  Re-use gradSquared for efficiency
            var small = 1.0e-05     //  Small offset to make sure we are not dividing by zero
            vDSP_vsaddD(gradSquared, 1, &small, &gradSquared, 1, vDSP_Length(numInputs))       //  (sqrt(rmsprop_cache) + 1e-5)
            var η = -averageTrainingRate     //  Needed for unsafe pointer conversion - negate for multiply-and-add vector operation
            vDSP_svdivD(&η, gradSquared, 1, &gradSquared, 1, vDSP_Length(numInputs))
            vDSP_vmaD(𝟃E𝟃W, 1, gradSquared, 1, W, 1, &W, 1, vDSP_Length(numInputs))
            
            //  Update the rmsProp cache for U --> rmsprop_cache = decay_rate * rmsprop_cache + (1 - decay_rate) * gradient²
            gradSquared = [Double](repeating: 0.0, count: numFeedback)
            vDSP_vsqD(𝟃E𝟃U, 1, &gradSquared, 1, vDSP_Length(numFeedback))  //  Get the gradient squared
            decay = 1.0 - weightUpdateParameter!
            vDSP_vsmulD(gradSquared, 1, &decay, &gradSquared, 1, vDSP_Length(numFeedback))   //  (1 - decay_rate) * gradient²
            decay = weightUpdateParameter!
            vDSP_vsmaD(UweightUpdateData, 1, &decay, gradSquared, 1, &UweightUpdateData, 1, vDSP_Length(numFeedback))
            //  Update the weights --> weight += learning_rate * gradient / (sqrt(rmsprop_cache) + 1e-5)
            for i in 0..<numFeedback { gradSquared[i] = sqrt(UweightUpdateData[i]) }      //  Re-use gradSquared for efficiency
            small = 1.0e-05     //  Small offset to make sure we are not dividing by zero
            vDSP_vsaddD(gradSquared, 1, &small, &gradSquared, 1, vDSP_Length(numFeedback))       //  (sqrt(rmsprop_cache) + 1e-5)
            η = -averageTrainingRate     //  Needed for unsafe pointer conversion - negate for multiply-and-add vector operation
            vDSP_svdivD(&η, gradSquared, 1, &gradSquared, 1, vDSP_Length(numFeedback))
            vDSP_vmaD(𝟃E𝟃U, 1, gradSquared, 1, U, 1, &U, 1, vDSP_Length(numFeedback))
        }
    }
    
    func decayWeights(_ decayFactor : Double)
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
    
    func gradientCheck(x: [Double], ε: Double, Δ: Double, network: NeuralNetwork)  -> Bool
    {
        var result = true
        
        //  Iterate through each W parameter
        for index in 0..<W.count {
            let oldValue = W[index]
            
            //  Get the network loss with a small addition to the parameter
            W[index] += ε
            _ = network.feedForward(x)
            var plusLoss : [Double]
            do {
                plusLoss = try network.getResultLoss()
            }
            catch {
                return false
            }
            
            //  Get the network loss with a small subtraction from the parameter
            W[index] = oldValue - ε
            _ = network.feedForward(x)
            var minusLoss : [Double]
            do {
                minusLoss = try network.getResultLoss()
            }
            catch {
                return false
            }
            W[index] = oldValue
            
            //  Iterate over the results
            for resultIndex in 0..<plusLoss.count {
                //  Get the numerical gradient estimate  𝟃E/𝟃W
                let gradient = (plusLoss[resultIndex] - minusLoss[resultIndex]) / (2.0 * ε)
                
                //  Compare with the analytical gradient
                let difference = abs(gradient - 𝟃E𝟃W[index])
                //                print("difference = \(difference)")
                if (difference > Δ) {
                    result = false
                }
            }
        }
        
        //  Iterate through each U parameter
        for index in 0..<U.count {
            let oldValue = U[index]
            
            //  Get the network loss with a small addition to the parameter
            U[index] += ε
            _ = network.feedForward(x)
            var plusLoss : [Double]
            do {
                plusLoss = try network.getResultLoss()
            }
            catch {
                return false
            }
            
            //  Get the network loss with a small subtraction from the parameter
            U[index] = oldValue - ε
            _ = network.feedForward(x)
            var minusLoss : [Double]
            do {
                minusLoss = try network.getResultLoss()
            }
            catch {
                return false
            }
            U[index] = oldValue
            
            //  Iterate over the results
            for resultIndex in 0..<plusLoss.count {
                //  Get the numerical gradient estimate  𝟃E/𝟃U
                let gradient = (plusLoss[resultIndex] - minusLoss[resultIndex]) / (2.0 * ε)
                
                //  Compare with the analytical gradient
                let difference = abs(gradient - 𝟃E𝟃U[index])
                //                print("difference = \(difference)")
                if (difference > Δ) {
                    result = false
                }
            }
        }
        
        return result
    }
}

///  Class for a recurrent network layer with individual nodes (slower, but easier to get into details)
final class RecurrentNeuralLayerWithNodes: NeuralLayer {
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
            weights += node.W
            weights += node.U
        }
        return weights
    }
    
    func setNeuralWeightUpdateMethod(_ method: NeuralWeightUpdateMethod, _ parameter: Double?)
    {
        for node in nodes {
            node.setNeuralWeightUpdateMethod(method, parameter)
        }
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
    
    func feedForward(_ x: [Double]) -> [Double]
    {
        //  Gather the previous outputs for the feedback
        var hPrev : [Double] = []
        for node in nodes {
            hPrev.append(node.getPreviousOutputValue())
        }
        
        var outputs : [Double] = []
        //  Assume input array already has bias constant 1.0 appended
        //  Fully-connected nodes means all nodes get the same input array
        if (nodes[0].activation == .softMax) {
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
    
    func getFinalLayer𝟃E𝟃zs(_ 𝟃E𝟃h: [Double])
    {
        for nNodeIndex in 0..<nodes.count {
            //  Start with the portion from the squared error term
            nodes[nNodeIndex].getFinalNode𝟃E𝟃zs(𝟃E𝟃h[nNodeIndex])
        }
    }
    
    func getLayer𝟃E𝟃zs(_ nextLayer: NeuralLayer)
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
    
    func get𝟃E𝟃hForNodeInPreviousLayer(_ inputIndex: Int) ->Double
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
    
    func appendWeightChanges(_ x: [Double]) -> [Double]
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
        if (activation == .hyperbolicTangent || activation == .rectifiedLinear) { return 0.0 }
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
    
    func retrieveRecurrentValues(_ sequenceIndex: Int)
    {
        bpttSequenceIndex =  sequenceIndex
        
        //  Set the last recurrent value in the history array to the last output
        for node in nodes {
            node.getLastRecurrentValue()
        }
    }
    
    func gradientCheck(x: [Double], ε: Double, Δ: Double, network: NeuralNetwork)  -> Bool
    {
        var result = true
        for node in nodes {
            if (!node.gradientCheck(x: x, ε: ε, Δ: Δ, network: network)) { result = false }
        }
        return result
    }
}


///  Class for a recurrent network layer without individual nodes (faster, but some things hidden in the matrix math)
final class RecurrentNeuralLayer: NeuralLayer {
    var activation : NeuralActivationFunction
    var numInputs = 0
    var numNodes : Int
    var W : [Double] = []      //  Weights for inputs from previous layer
    var U : [Double] = []      //  Weights for recurrent input data from this layer
    var h : [Double] //  Last result calculated
    var outputHistory : [[Double]] //  History of output for the sequence
    var 𝟃E𝟃z : [Double]      //  Gradient in error with respect to weighted sum
    var 𝟃E𝟃W : [Double] = []  //  Accumulated weight W change gradient
    var 𝟃E𝟃U : [Double] = []  //  Accumulated weight U change gradient
    var bpttSequenceIndex : Int
    var weightUpdateMethod = NeuralWeightUpdateMethod.normal
    var weightUpdateParameter : Double?      //  Decay rate for rms prop weight updates
    var WweightUpdateData : [Double] = []    //  Array of running average for rmsprop
    var UweightUpdateData : [Double] = []    //  Array of running average for rmsprop

    ///  Create the neural network layer based on a tuple (number of nodes, activation function)
    init(numInputs : Int, layerDefinition: (layerType: NeuronLayerType, numNodes: Int, activation: NeuralActivationFunction, auxiliaryData: AnyObject?))
    {
        activation = layerDefinition.activation
        self.numInputs = numInputs
        self.numNodes = layerDefinition.numNodes
        h = [Double](repeating: 0.0, count: numNodes)
        outputHistory = []
        𝟃E𝟃z = [Double](repeating: 0.0, count: numNodes)
        bpttSequenceIndex = 0
    }
    
    //  Initialize the weights
    func initWeights(_ startWeights: [Double]!)
    {
        let numWeights = (numInputs + 1) * numNodes   //  Add bias offset
        let numRcurrentWeights = numNodes * numNodes   //  Add bias offset
        W = []
        U = []
        if let startWeights = startWeights {
            if (startWeights.count >= numNodes * (numInputs + 1 + numNodes)) {
                //  If there are enough weights for all nodes, split the weights and initialize
                W = Array(startWeights[0..<(numNodes * (numInputs + 1))])
                U = Array(startWeights[(numNodes * (numInputs + 1))..<(numNodes * (numInputs + 1 + numNodes))])
            }
            else {
                //  If there are not enough weights for all nodes, initialize each weight set with the set given
                var index = 0
                for _ in 0..<((numInputs + 1) * numNodes) {
                    W.append(startWeights[index])
                    index += 1
                    if (index >= startWeights.count) { index = 0 }
                }
                for _ in 0..<(numNodes * numNodes) {
                    U.append(startWeights[index])
                    index += 1
                    if (index >= startWeights.count) { index = 0 }
                }
            }
        }
        else {
            //  No specified weights - just initialize normally
            //  Allocate the weight array using 'Xavier' initialization
            var weightDiviser: Double
            if (activation == .rectifiedLinear) {
                weightDiviser = 1 / sqrt(Double(numInputs) * 0.5)
            }
            else {
                weightDiviser = 1 / sqrt(Double(numInputs))
            }
            for _ in 0..<numWeights {
                W.append(Gaussian.gaussianRandom(0.0, standardDeviation : 1.0) * weightDiviser)
            }
            if (activation == .rectifiedLinear) {
                weightDiviser = 1 / sqrt(Double(numNodes) * 0.5)
            }
            else {
                weightDiviser = 1 / sqrt(Double(numNodes))
            }
            for _ in 0..<numRcurrentWeights {
                U.append(Gaussian.gaussianRandom(0.0, standardDeviation : 1.0) * weightDiviser)
            }
        }
        
        //  If rmsprop update, allocate the momentum storage array
        if (weightUpdateMethod == .rmsProp) {
            WweightUpdateData = [Double](repeating: 0.0, count: numWeights)
            UweightUpdateData = [Double](repeating: 0.0, count: numRcurrentWeights)
        }
    }
    
    func getWeights() -> [Double]
    {
        var weights = W
        weights += U
        return weights
    }
    
    func setNeuralWeightUpdateMethod(_ method: NeuralWeightUpdateMethod, _ parameter: Double?)
    {
        weightUpdateMethod = method
        weightUpdateParameter = parameter
    }
    
    func getLastOutput() -> [Double]
    {
        return h
    }
    
    func getNodeCount() -> Int
    {
        return numNodes
    }
    
    func getWeightsPerNode()-> Int
    {
        return numInputs + numNodes + 1
    }
    
    func getActivation()-> NeuralActivationFunction
    {
        return activation
    }
    
    func feedForward(_ x: [Double]) -> [Double]
    {
        //  Gather the previous outputs for the feedback
        var hPrev : [Double] = []
        if let temp = outputHistory.last {
            hPrev = temp
        }
        else {
            hPrev = [Double](repeating: 0.0, count: numNodes)
        }
        
        var z = [Double](repeating: 0.0, count: numNodes)
        var uz = [Double](repeating: 0.0, count: numNodes)
        //  Assume input array already has bias constant 1.0 appended
        //  Fully-connected nodes means all nodes get the same input array
        vDSP_mmulD(W, 1, x, 1, &z, 1, vDSP_Length(numNodes), 1, vDSP_Length(numInputs+1))
        vDSP_mmulD(U, 1, hPrev, 1, &uz, 1, vDSP_Length(numNodes), 1, vDSP_Length(numNodes))
        vDSP_vaddD(z, 1, uz, 1, &z, 1, vDSP_Length(numNodes))
        
        //  Run through the non-linearity
        var sum = 0.0
        for node in 0..<numNodes {
            switch (activation) {
            case .none:
                h[node] = z[node]
                break
            case .hyperbolicTangent:
                h[node] = tanh(z[node])
                break
            case .sigmoidWithCrossEntropy:
                h[node] = 1.0 / (1.0 + exp(-z[node]))
                sum += h[node]
                break
            case .sigmoid:
                h[node] = 1.0 / (1.0 + exp(-z[node]))
                break
            case .rectifiedLinear:
                h[node] = z[node]
                if (z[node] < 0) { h[node] = 0.0 }
                break
            case .softSign:
                h[node] = z[node] / (1.0 + abs(z[node]))
                break
            case .softMax:
                h[node] = exp(z[node])
                break
            }
        }
        
        if (activation == .softMax) {
            var scale = 1.0 / sum       //  Do division once for efficiency
            vDSP_vsmulD(h, 1, &scale, &h, 1, vDSP_Length(numNodes))
        }
        
        return h
    }
    
    func getFinalLayer𝟃E𝟃zs(_ 𝟃E𝟃h: [Double])
    {
        //  Calculate 𝟃E/𝟃z from 𝟃E/𝟃h
        switch (activation) {
        case .none:
            𝟃E𝟃z = 𝟃E𝟃h
            break
        case .hyperbolicTangent:
            vDSP_vsqD(h, 1, &𝟃E𝟃z, 1, vDSP_Length(numNodes))       //  h²
            let ones = [Double](repeating: 1.0, count: numNodes)
            vDSP_vsubD(𝟃E𝟃z, 1, ones, 1, &𝟃E𝟃z, 1, vDSP_Length(numNodes))       //  1 - h²
            vDSP_vmulD(𝟃E𝟃z, 1, 𝟃E𝟃h, 1, &𝟃E𝟃z, 1, vDSP_Length(numNodes))       //  𝟃E𝟃h * (1 - h²)
            break
        case .sigmoidWithCrossEntropy:
            fallthrough
        case .sigmoid:
            vDSP_vsqD(h, 1, &𝟃E𝟃z, 1, vDSP_Length(numNodes))       //  h²
            vDSP_vsubD(𝟃E𝟃z, 1, h, 1, &𝟃E𝟃z, 1, vDSP_Length(numNodes))       //  h - h²
            vDSP_vmulD(𝟃E𝟃z, 1, 𝟃E𝟃h, 1, &𝟃E𝟃z, 1, vDSP_Length(numNodes))       //  𝟃E𝟃h * (h - h²)
            break
        case .rectifiedLinear:
            for i in 0..<numNodes {
                𝟃E𝟃z[i] = h[i] <= 0.0 ? 0.0 : 𝟃E𝟃h[i]
            }
            break
        case .softSign:
            for i in 0..<numNodes {
                //  Reconstitute z from h
                var z : Double
                //!! - this might be able to be sped up with vector operations
                if (h[i] < 0) {        //  Negative z
                    z = h[i] / (1.0 + h[i])
                    𝟃E𝟃z[i] = -𝟃E𝟃h[i] / ((1.0 + z) * (1.0 + z))
                }
                else {              //  Positive z
                    z = h[i] / (1.0 - h[i])
                    𝟃E𝟃z[i] = 𝟃E𝟃h[i] / ((1.0 + z) * (1.0 + z))
                }
            }
            break
        case .softMax:
            //  This should be done outside of the layer
            break
        }
    }
    
    func getLayer𝟃E𝟃zs(_ nextLayer: NeuralLayer)
    {
        //  Get 𝟃E/𝟃h from the next layer
        var 𝟃E𝟃h = [Double](repeating: 0.0, count: numNodes)
        for node in 0..<numNodes {
            𝟃E𝟃h[node] = nextLayer.get𝟃E𝟃hForNodeInPreviousLayer(node)
        }
        
        //  Calculate 𝟃E/𝟃z from 𝟃E/𝟃h
        getFinalLayer𝟃E𝟃zs(𝟃E𝟃h)
    }
    
    func get𝟃E𝟃hForNodeInPreviousLayer(_ inputIndex: Int) ->Double
    {
        var sum = 0.0
        var offset = inputIndex
        for node in 0..<numNodes {
            sum += 𝟃E𝟃z[node] * W[offset]
            offset += numInputs+1
        }
        return sum
    }
    
    func clearWeightChanges()
    {
        𝟃E𝟃W = [Double](repeating: 0.0, count: W.count)
        𝟃E𝟃U = [Double](repeating: 0.0, count: U.count)
    }
    
    func appendWeightChanges(_ x: [Double]) -> [Double]
    {
        //  Gather the previous outputs for the feedback
        var hPrev : [Double] = []
        if let temp = outputHistory.last {
            hPrev = temp
        }
        else {
            hPrev = [Double](repeating: 0.0, count: numNodes)
        }
        
        //  Assume input array already has bias constant 1.0 appended
        //  Update each weight accumulation
        var weightWChange = [Double](repeating: 0.0, count: W.count)
        vDSP_mmulD(𝟃E𝟃z, 1, x, 1, &weightWChange, 1, vDSP_Length(numNodes), vDSP_Length(numInputs+1), 1)
        vDSP_vaddD(weightWChange, 1, 𝟃E𝟃W, 1, &𝟃E𝟃W, 1, vDSP_Length(W.count))
        var weightUChange = [Double](repeating: 0.0, count: U.count)
        vDSP_mmulD(𝟃E𝟃z, 1, hPrev, 1, &weightUChange, 1, vDSP_Length(numNodes), vDSP_Length(numNodes), 1)
        vDSP_vaddD(weightUChange, 1, 𝟃E𝟃U, 1, &𝟃E𝟃U, 1, vDSP_Length(U.count))
        
        return h
    }
    
    func updateWeightsFromAccumulations(_ averageTrainingRate: Double, weightDecay: Double)
    {
        //  Decay weights if indicated
        if (weightDecay < 1) { decayWeights(weightDecay) }
        
        //  Update the weights from the accumulations
        switch weightUpdateMethod {
        case .normal:
            //  W -= 𝟃E/𝟃W * averageTrainingRate, U -= 𝟃E/𝟃U * averageTrainingRate
            var η = -averageTrainingRate     //  Needed for unsafe pointer conversion - negate for multiply-and-add vector operation
            vDSP_vsmaD(𝟃E𝟃W, 1, &η, W, 1, &W, 1, vDSP_Length(W.count))
            vDSP_vsmaD(𝟃E𝟃U, 1, &η, U, 1, &U, 1, vDSP_Length(U.count))
        case .rmsProp:
            //  Update the rmsProp cache for W --> rmsprop_cache = decay_rate * rmsprop_cache + (1 - decay_rate) * gradient²
            let numWeights = W.count
            var gradSquared = [Double](repeating: 0.0, count: numWeights)
            vDSP_vsqD(𝟃E𝟃W, 1, &gradSquared, 1, vDSP_Length(numWeights))  //  Get the gradient squared
            var decay = 1.0 - weightUpdateParameter!
            vDSP_vsmulD(gradSquared, 1, &decay, &gradSquared, 1, vDSP_Length(numWeights))   //  (1 - decay_rate) * gradient²
            decay = weightUpdateParameter!
            vDSP_vsmaD(WweightUpdateData, 1, &decay, gradSquared, 1, &WweightUpdateData, 1, vDSP_Length(numWeights))
            //  Update the weights --> weight += learning_rate * gradient / (sqrt(rmsprop_cache) + 1e-5)
            for i in 0..<numWeights { gradSquared[i] = sqrt(WweightUpdateData[i]) }      //  Re-use gradSquared for efficiency
            var small = 1.0e-05     //  Small offset to make sure we are not dividing by zero
            vDSP_vsaddD(gradSquared, 1, &small, &gradSquared, 1, vDSP_Length(numWeights))       //  (sqrt(rmsprop_cache) + 1e-5)
            var η = -averageTrainingRate     //  Needed for unsafe pointer conversion - negate for multiply-and-add vector operation
            vDSP_svdivD(&η, gradSquared, 1, &gradSquared, 1, vDSP_Length(numWeights))
            vDSP_vmaD(𝟃E𝟃W, 1, gradSquared, 1, W, 1, &W, 1, vDSP_Length(numWeights))
            
            //  Update the rmsProp cache for U --> rmsprop_cache = decay_rate * rmsprop_cache + (1 - decay_rate) * gradient²
            let numFeedback = U.count
            gradSquared = [Double](repeating: 0.0, count: numFeedback)
            vDSP_vsqD(𝟃E𝟃U, 1, &gradSquared, 1, vDSP_Length(numFeedback))  //  Get the gradient squared
            decay = 1.0 - weightUpdateParameter!
            vDSP_vsmulD(gradSquared, 1, &decay, &gradSquared, 1, vDSP_Length(numFeedback))   //  (1 - decay_rate) * gradient²
            decay = weightUpdateParameter!
            vDSP_vsmaD(UweightUpdateData, 1, &decay, gradSquared, 1, &UweightUpdateData, 1, vDSP_Length(numFeedback))
            //  Update the weights --> weight += learning_rate * gradient / (sqrt(rmsprop_cache) + 1e-5)
            for i in 0..<numFeedback { gradSquared[i] = sqrt(UweightUpdateData[i]) }      //  Re-use gradSquared for efficiency
            small = 1.0e-05     //  Small offset to make sure we are not dividing by zero
            vDSP_vsaddD(gradSquared, 1, &small, &gradSquared, 1, vDSP_Length(numFeedback))       //  (sqrt(rmsprop_cache) + 1e-5)
            η = -averageTrainingRate     //  Needed for unsafe pointer conversion - negate for multiply-and-add vector operation
            vDSP_svdivD(&η, gradSquared, 1, &gradSquared, 1, vDSP_Length(numFeedback))
            vDSP_vmaD(𝟃E𝟃U, 1, gradSquared, 1, U, 1, &U, 1, vDSP_Length(numFeedback))
        }
    }
    
    func decayWeights(_ decayFactor : Double)
    {
        var decay = decayFactor
        vDSP_vsmulD(W, 1, &decay, &W, 1, vDSP_Length(W.count))
        vDSP_vsmulD(U, 1, &decay, &U, 1, vDSP_Length(U.count))
    }
    
    func getSingleNodeClassifyValue() -> Double
    {
        if (activation == .hyperbolicTangent || activation == .rectifiedLinear) { return 0.0 }
        return 0.5
    }
    
    func resetSequence()
    {
        //  Have each node reset
        h = [Double](repeating: 0.0, count: numNodes)
        outputHistory = [[Double](repeating: 0.0, count: numNodes)]       //  first 'previous' value is zero
        𝟃E𝟃z = [Double](repeating: 0.0, count: numNodes)                 //  Backward propogation previous 𝟃E𝟃z (𝟃E𝟃z from next time step in sequence) is zero
    }
    
    func storeRecurrentValues()
    {
        outputHistory.append(h)
    }
    
    func retrieveRecurrentValues(_ sequenceIndex: Int)
    {
        bpttSequenceIndex =  sequenceIndex
        
        //  Set the last recurrent value in the history array to the last output
        h = outputHistory.removeLast()
    }
    
    func gradientCheck(x: [Double], ε: Double, Δ: Double, network: NeuralNetwork)  -> Bool
    {
        var result = true
        
        //  Iterate through each W parameter
        for index in 0..<W.count {
            let oldValue = W[index]
            
            //  Get the network loss with a small addition to the parameter
            W[index] += ε
            _ = network.feedForward(x)
            var plusLoss : [Double]
            do {
                plusLoss = try network.getResultLoss()
            }
            catch {
                return false
            }
            
            //  Get the network loss with a small subtraction from the parameter
            W[index] = oldValue - ε
            _ = network.feedForward(x)
            var minusLoss : [Double]
            do {
                minusLoss = try network.getResultLoss()
            }
            catch {
                return false
            }
            W[index] = oldValue
            
            //  Iterate over the results
            for resultIndex in 0..<plusLoss.count {
                //  Get the numerical gradient estimate  𝟃E/𝟃W
                let gradient = (plusLoss[resultIndex] - minusLoss[resultIndex]) / (2.0 * ε)
                
                //  Compare with the analytical gradient
                let difference = abs(gradient - 𝟃E𝟃W[index])
                //                print("difference = \(difference)")
                if (difference > Δ) {
                    result = false
                }
            }
        }
        
        //  Iterate through each U parameter
        for index in 0..<U.count {
            let oldValue = U[index]
            
            //  Get the network loss with a small addition to the parameter
            U[index] += ε
            _ = network.feedForward(x)
            var plusLoss : [Double]
            do {
                plusLoss = try network.getResultLoss()
            }
            catch {
                return false
            }
            
            //  Get the network loss with a small subtraction from the parameter
            U[index] = oldValue - ε
            _ = network.feedForward(x)
            var minusLoss : [Double]
            do {
                minusLoss = try network.getResultLoss()
            }
            catch {
                return false
            }
            U[index] = oldValue
            
            //  Iterate over the results
            for resultIndex in 0..<plusLoss.count {
                //  Get the numerical gradient estimate  𝟃E/𝟃U
                let gradient = (plusLoss[resultIndex] - minusLoss[resultIndex]) / (2.0 * ε)
                
                //  Compare with the analytical gradient
                let difference = abs(gradient - 𝟃E𝟃U[index])
                //                print("difference = \(difference)")
                if (difference > Δ) {
                    result = false
                }
            }
        }
        
        return result
    }
}

