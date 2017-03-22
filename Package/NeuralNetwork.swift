//
//  NeuralNetwork.swift
//  AIToolbox
//
//  Created by Kevin Coble on 10/6/15.
//  Copyright © 2015 Kevin Coble. All rights reserved.
//

import Foundation
#if os(Linux)
#else
import Accelerate
#endif

public enum NeuronLayerType {
    case simpleFeedForwardWithNodes
    case simpleFeedForward
    case simpleRecurrentWithNodes
    case simpleRecurrent
    case lstm
}

public enum NeuralActivationFunction : Int {
    case none=0
    case hyperbolicTangent
    case sigmoid
    case sigmoidWithCrossEntropy
    case rectifiedLinear
    case softSign
    case softMax        //  Only valid on output (last) layer
    
    func getString() ->String
    {
        switch self {
        case .none:
            return "None"
        case .hyperbolicTangent:
            return "tanh"
        case .sigmoid:
            return "Sigmoid"
        case .sigmoidWithCrossEntropy:
            return "Sigmoid with X-entropy"
        case .rectifiedLinear:
            return "Rect. Linear"
        case .softSign:
            return "Soft-sign"
        case .softMax:
            return "Soft-Max"
        }
    }
}

public enum NeuralWeightUpdateMethod : Int {
    case normal = 0
    case rmsProp
}

protocol NeuralLayer {
    func getNodeCount() -> Int
    func getWeightsPerNode()-> Int
    func getActivation()-> NeuralActivationFunction
    func feedForward(_ x: [Double]) -> [Double]
    func initWeights(_ startWeights: [Double]!)
    func getWeights() -> [Double]
    func setNeuralWeightUpdateMethod(_ method: NeuralWeightUpdateMethod, _ parameter: Double?)
    func getLastOutput() -> [Double]
    func getFinalLayer𝟃E𝟃zs(_ 𝟃E𝟃h: [Double])
    func getLayer𝟃E𝟃zs(_ nextLayer: NeuralLayer)
    func get𝟃E𝟃hForNodeInPreviousLayer(_ inputIndex: Int) ->Double
    func clearWeightChanges()
    func appendWeightChanges(_ inputs: [Double]) -> [Double]
    func updateWeightsFromAccumulations(_ averageTrainingRate: Double, weightDecay: Double)
    func decayWeights(_ decayFactor : Double)
    func getSingleNodeClassifyValue() -> Double
    func resetSequence()
    func storeRecurrentValues()
    func retrieveRecurrentValues(_ sequenceIndex: Int)
    func gradientCheck(x: [Double], ε: Double, Δ: Double, network: NeuralNetwork) -> Bool
}

public enum NeuralNetworkError: Error {
    case expectedOutputNotSet
}

final class SimpleNeuralNode {
    //  Activation function
    let activation : NeuralActivationFunction
    let numWeights : Int        //  Includes bias weight
    var W : [Double]      //  Weight vector
    var h : Double //  Last result calculated
    var outputHistory : [Double] //  History of output for the sequence
    var 𝟃E𝟃h : Double      //  Gradient in error with respect to output of this node
    var 𝟃E𝟃z : Double      //  Gradient in error with respect to weighted sum
    var 𝟃E𝟃W : [Double]   //  Accumulated weight change gradient
    var weightUpdateMethod = NeuralWeightUpdateMethod.normal
    var weightUpdateParameter : Double?      //  Decay rate for rms prop weight updates
    var weightUpdateData : [Double] = []    //  Array of running average for rmsprop
    
    ///  Create the neural network node with a set activation function
    init(numInputs : Int, activationFunction: NeuralActivationFunction)
    {
        activation = activationFunction
        numWeights = numInputs + 1  //  Add one weight for the bias term
        W = []
        h = 0.0
        outputHistory = []
        𝟃E𝟃h = 0.0
        𝟃E𝟃z = 0.0
        𝟃E𝟃W = []
    }
    
    //  Initialize the weights
    func initWeights(_ startWeights: [Double]!)
    {
        if let startWeights = startWeights {
            if (startWeights.count == 1) {
                W = [Double](repeating: startWeights[0], count: numWeights)
            }
            else {
                W = []
                var index = 0 //  Last number (if more than 1) goes into the bias weight, then repeat the ones that come before
                for _ in 0..<numWeights-1  {
                    if (index >= startWeights.count-1) { index = 0 }      //  Wrap if necessary
                    W.append(startWeights[index])
                    index += 1
                }
                W.append(startWeights[startWeights.count-1])     //  Add the bias term
            }
        }
        else {
            W = []
            for _ in 0..<numWeights-1  {
                W.append(Gaussian.gaussianRandom(0.0, standardDeviation: 1.0 / Double(numWeights-1)))    //  input weights - Initialize to a random number to break initial symmetry of the network, scaled to the inputs
            }
            W.append(Gaussian.gaussianRandom(0.0, standardDeviation:1.0))    //  Bias weight - Initialize to a  random number to break initial symmetry of the network
        }
        
        //  If rmsprop update, allocate the momentum storage array
        if (weightUpdateMethod == .rmsProp) {
            weightUpdateData = [Double](repeating: 0.0, count: numWeights)
        }
    }
    
    func setNeuralWeightUpdateMethod(_ method: NeuralWeightUpdateMethod, _ parameter: Double?)
    {
        weightUpdateMethod = method
        weightUpdateParameter = parameter
    }
    
    func feedForward(_ x: [Double]) -> Double
    {
        //  Get the weighted sum:  z = W⋅x
        var z = 0.0
        vDSP_dotprD(W, 1, x, 1, &z, vDSP_Length(W.count))
        
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
    
    func get𝟃E𝟃z()
    {
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
                //  Should not get here - SoftMax is only valid on output layer
                break
        }
    }
    
    func clearWeightChanges()
    {
        𝟃E𝟃W = [Double](repeating: 0.0, count: W.count)
    }
    
    func appendWeightChanges(_ x: [Double]) -> Double
    {
        //  Update each weight accumulation
        //  𝟃E/𝟃W = 𝟃E/𝟃z ⋅ 𝟃z/𝟃W = 𝟃E/𝟃z ⋅ x
        vDSP_vsmaD(x, 1, &𝟃E𝟃z, 𝟃E𝟃W, 1, &𝟃E𝟃W, 1, vDSP_Length(numWeights))
        
        return h    //  return output for next layer
    }
    
    func updateWeightsFromAccumulations(_ averageTrainingRate: Double)
    {
        //  Update the weights from the accumulations
        switch weightUpdateMethod {
        case .normal:
            //  W -= 𝟃E/𝟃W * averageTrainingRate
            var η = -averageTrainingRate     //  Needed for unsafe pointer conversion - negate for multiply-and-add vector operation
            vDSP_vsmaD(𝟃E𝟃W, 1, &η, W, 1, &W, 1, vDSP_Length(numWeights))
        case .rmsProp:
            //  Update the rmsProp cache --> rmsprop_cache = decay_rate * rmsprop_cache + (1 - decay_rate) * gradient²
            var gradSquared = [Double](repeating: 0.0, count: numWeights)
            vDSP_vsqD(𝟃E𝟃W, 1, &gradSquared, 1, vDSP_Length(numWeights))  //  Get the gradient squared
            var decay = 1.0 - weightUpdateParameter!
            vDSP_vsmulD(gradSquared, 1, &decay, &gradSquared, 1, vDSP_Length(numWeights))   //  (1 - decay_rate) * gradient²
            decay = weightUpdateParameter!
            vDSP_vsmaD(weightUpdateData, 1, &decay, gradSquared, 1, &weightUpdateData, 1, vDSP_Length(numWeights))
            //  Update the weights --> weight += learning_rate * gradient / (sqrt(rmsprop_cache) + 1e-5)
            for i in 0..<numWeights { gradSquared[i] = sqrt(weightUpdateData[i]) }      //  Re-use gradSquared for efficiency
            var small = 1.0e-05     //  Small offset to make sure we are not dividing by zero
            vDSP_vsaddD(gradSquared, 1, &small, &gradSquared, 1, vDSP_Length(numWeights))       //  (sqrt(rmsprop_cache) + 1e-5)
            var η = -averageTrainingRate     //  Needed for unsafe pointer conversion - negate for multiply-and-add vector operation
            vDSP_svdivD(&η, gradSquared, 1, &gradSquared, 1, vDSP_Length(numWeights))
            vDSP_vmaD(𝟃E𝟃W, 1, gradSquared, 1, W, 1, &W, 1, vDSP_Length(numWeights))
        }
    }
    
    func decayWeights(_ decayFactor : Double)
    {
        var λ = decayFactor     //  Needed for unsafe pointer conversion
        vDSP_vsmulD(W, 1, &λ, &W, 1, vDSP_Length(numWeights-1))
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
    
    func gradientCheck(x: [Double], ε: Double, Δ: Double, network: NeuralNetwork)  -> Bool
    {
        var result = true
        
        //  Iterate through each parameter
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
        
        return result
    }
}

///  Class for a feed-forward network with individual nodes (slower, but easier to get into details)
final class SimpleNeuralLayerWithNodes: NeuralLayer {
    //  Nodes
    var nodes : [SimpleNeuralNode]
    
    ///  Create the neural network layer based on a tuple (number of nodes, activation function)
    init(numInputs : Int, layerDefinition: (layerType: NeuronLayerType, numNodes: Int, activation: NeuralActivationFunction, auxiliaryData: AnyObject?))
    {
        nodes = []
        for _ in 0..<layerDefinition.numNodes {
            nodes.append(SimpleNeuralNode(numInputs: numInputs, activationFunction: layerDefinition.activation))
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
            weights += node.W
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
        var outputs : [Double] = []
        //  Assume input array already has bias constant 1.0 appended
        //  Fully-connected nodes means all nodes get the same input array
        if (nodes[0].activation == .softMax) {
            var sum = 0.0
            for node in nodes {     //  Sum each output
                sum += node.feedForward(x)
            }
            let scale = 1.0 / sum       //  Do division once for efficiency
            for node in nodes {     //  Get the outputs scaled by the sum to give the probability distribuition for the output
                node.h *= scale
                outputs.append(node.h)
            }
        }
        else {
            for node in nodes {
                outputs.append(node.feedForward(x))
            }
        }
        
        return outputs
    }
    
    func getFinalLayer𝟃E𝟃zs(_ 𝟃E𝟃h: [Double])
    {
        for nodeIndex in 0..<nodes.count {
            //  Set 𝟃E/𝟃h from external error
            nodes[nodeIndex].𝟃E𝟃h = 𝟃E𝟃h[nodeIndex]
            
            //  Backpropogate error to the z value level
            nodes[nodeIndex].get𝟃E𝟃z()
        }
    }
    
    func getLayer𝟃E𝟃zs(_ nextLayer: NeuralLayer)
    {
        for nNodeIndex in 0..<nodes.count {
            //  Set 𝟃E/𝟃h to 0
            nodes[nNodeIndex].reset𝟃E𝟃hs()
            
            //  Add each portion of 𝟃E/𝟃h from the nodes in the next forward layer
            nodes[nNodeIndex].addTo𝟃E𝟃hs(nextLayer.get𝟃E𝟃hForNodeInPreviousLayer(nNodeIndex))
            
            //  Calculate 𝟃E/𝟃z from 𝟃E/𝟃h
            nodes[nNodeIndex].get𝟃E𝟃z()
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
        var outputs : [Double] = []
        //  Assume input array already has bias constant 1.0 appended
        //  Fully-connected nodes means all nodes get the same input array
        for node in nodes {
            outputs.append(node.appendWeightChanges(x))
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
        //  Set the last recurrent value in the history array to the last output
        for node in nodes {
            node.getLastRecurrentValue()
        }
    }
    
    func gradientCheck(x: [Double], ε: Double, Δ: Double, network: NeuralNetwork)  -> Bool
    {
        //  Have each node check it's own gradients
        var result = true
        for node in nodes {
            if (!node.gradientCheck(x: x, ε: ε, Δ: Δ, network: network)) { result = false }
        }
        return result
    }
}

///  Class for a feed-forward network without individual nodes (faster, but details hidden in matrices)
final class SimpleNeuralLayer: NeuralLayer {
    var activation : NeuralActivationFunction
    var numInputs = 0
    var numNodes : Int
    var W : [Double] = []
    var h : [Double] //  Last result calculated
    var outputHistory : [[Double]] //  History of output for the sequence
    var 𝟃E𝟃z : [Double]      //  Gradient in error with respect to weighted sum
    var 𝟃E𝟃W : [Double] = []  //  Accumulated weight change gradient
    var weightUpdateMethod = NeuralWeightUpdateMethod.normal
    var weightUpdateParameter : Double?      //  Decay rate for rms prop weight updates
    var weightUpdateData : [Double] = []    //  Array of running average for rmsprop
    
    ///  Create the neural network layer based on a tuple (number of nodes, activation function)
    init(numInputs : Int, layerDefinition: (layerType: NeuronLayerType, numNodes: Int, activation: NeuralActivationFunction, auxiliaryData: AnyObject?))
    {
        activation = layerDefinition.activation
        self.numInputs = numInputs
        self.numNodes = layerDefinition.numNodes
        h = [Double](repeating: 0.0, count: numNodes)
        outputHistory = []
        𝟃E𝟃z = [Double](repeating: 0.0, count: numNodes)
    }
    
    //  Initialize the weights
    func initWeights(_ startWeights: [Double]!)
    {
        let numWeights = (numInputs + 1) * numNodes   //  Add bias offset
        W = []
        if let startWeights = startWeights {
            if (startWeights.count >= numWeights) {
                //  If there are enough weights for all nodes, just initialize
                for i in 0..<numWeights {
                    W.append(startWeights[i])
                }
            }
            else {
                //  If there are not enough weights for all nodes, initialize each (virtual) node with the set given
                for _ in 0..<numNodes {
                    for i in 0..<(numInputs+1) {
                        W.append(startWeights[i])
                    }
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
            W = []
            for _ in 0..<numWeights {
                W.append(Gaussian.gaussianRandom(0.0, standardDeviation : 1.0) * weightDiviser)
            }
        }
        
        //  If rmsprop update, allocate the momentum storage array
        if (weightUpdateMethod == .rmsProp) {
            weightUpdateData = [Double](repeating: 0.0, count: W.count)
        }
    }
    
    func getWeights() -> [Double]
    {
        return W
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
        return numInputs + 1    //  include bias
    }
    
    func getActivation()-> NeuralActivationFunction
    {
        return activation
    }
    
    func feedForward(_ x: [Double]) -> [Double]
    {
        var z = [Double](repeating: 0.0, count: numNodes)
        //  Assume input array already has bias constant 1.0 appended
        //  Fully-connected nodes means all nodes get the same input array
        vDSP_mmulD(W, 1, x, 1, &z, 1, vDSP_Length(numNodes), 1, vDSP_Length(numInputs+1))
        
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
    }
    
    func appendWeightChanges(_ x: [Double]) -> [Double]     //  Assumes x has bias term appended
    {
        //  Update each weight accumulation
        //  𝟃E/𝟃W = 𝟃E/𝟃z ⋅ 𝟃z/𝟃W = 𝟃E/𝟃z ⋅ x
        var weightChange = [Double](repeating: 0.0, count: W.count)
        vDSP_mmulD(𝟃E𝟃z, 1, x, 1, &weightChange, 1, vDSP_Length(numNodes), vDSP_Length(numInputs+1), 1)
        vDSP_vaddD(weightChange, 1, 𝟃E𝟃W, 1, &𝟃E𝟃W, 1, vDSP_Length(W.count))
        
        return h    //  return output for next layer
    }
    
    func updateWeightsFromAccumulations(_ averageTrainingRate: Double, weightDecay: Double)
    {
        //  Decay the weights if indicated
        if (weightDecay < 1) { decayWeights(weightDecay) }

        //  Update the weights from the accumulations
        let numWeights = W.count
        switch weightUpdateMethod {
        case .normal:
            //  W -= 𝟃E/𝟃W * averageTrainingRate
            var η = -averageTrainingRate     //  Needed for unsafe pointer conversion - negate for multiply-and-add vector operation
            vDSP_vsmaD(𝟃E𝟃W, 1, &η, W, 1, &W, 1, vDSP_Length(numWeights))
        case .rmsProp:
            //  Update the rmsProp cache --> rmsprop_cache = decay_rate * rmsprop_cache + (1 - decay_rate) * gradient²
            var gradSquared = [Double](repeating: 0.0, count: numWeights)
            vDSP_vsqD(𝟃E𝟃W, 1, &gradSquared, 1, vDSP_Length(numWeights))  //  Get the gradient squared
            var decay = 1.0 - weightUpdateParameter!
            vDSP_vsmulD(gradSquared, 1, &decay, &gradSquared, 1, vDSP_Length(numWeights))   //  (1 - decay_rate) * gradient²
            decay = weightUpdateParameter!
            vDSP_vsmaD(weightUpdateData, 1, &decay, gradSquared, 1, &weightUpdateData, 1, vDSP_Length(numWeights))
            //  Update the weights --> weight += learning_rate * gradient / (sqrt(rmsprop_cache) + 1e-5)
            for i in 0..<numWeights { gradSquared[i] = sqrt(weightUpdateData[i]) }      //  Re-use gradSquared for efficiency
            var small = 1.0e-05     //  Small offset to make sure we are not dividing by zero
            vDSP_vsaddD(gradSquared, 1, &small, &gradSquared, 1, vDSP_Length(numWeights))       //  (sqrt(rmsprop_cache) + 1e-5)
            var η = -averageTrainingRate     //  Needed for unsafe pointer conversion - negate for multiply-and-add vector operation
            vDSP_svdivD(&η, gradSquared, 1, &gradSquared, 1, vDSP_Length(numWeights))
            vDSP_vmaD(𝟃E𝟃W, 1, gradSquared, 1, W, 1, &W, 1, vDSP_Length(numWeights))
        }
}
    
    func decayWeights(_ decayFactor : Double)
    {
        var decay = decayFactor
        vDSP_vsmulD(W, 1, &decay, &W, 1, vDSP_Length(W.count))
    }
    
    func getSingleNodeClassifyValue() -> Double
    {
        if (activation == .hyperbolicTangent || activation == .rectifiedLinear) { return 0.0 }
        return 0.5
    }
    
    func resetSequence()
    {
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
        //  Set the last recurrent value in the history array to the last output
        h = outputHistory.removeLast()
    }
    
    func gradientCheck(x: [Double], ε: Double, Δ: Double, network: NeuralNetwork)  -> Bool
    {
        var result = true
        
        //  Iterate through each parameter
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
        
        return result
    }
}

open class NeuralNetwork: Classifier, Regressor {
    //  Layers
    let numInputs : Int
    let numOutputs : Int
    var layers : [NeuralLayer]
    var trainingRate = 0.3
    var weightDecay = 1.0
    var initializeFunction : ((_ trainData: MLDataSet)->[Double])!
    var hasRecurrentLayers = false
    var expectedOutput : [Double]?      //  Expected output for gradient checks
    public var lastClassificationOutput : [Double]?
    
    
    ///  Create the neural network based on an array of tuples, one for each non-input layer (number of nodes, activation function)
    ///     The input layer is defined by the number of inputs only.  The network is fully connected, including a bias term
    ///     If being used for classification, have the output (last) layer have the number of nodes equal to the number of classes
    public init(numInputs : Int, layerDefinitions: [(layerType: NeuronLayerType, numNodes: Int, activation: NeuralActivationFunction, auxiliaryData: AnyObject?)])
    {
        self.numInputs = numInputs
        layers = []
        var numInputsFromPreviousLayer = numInputs
        for layerDefinition in layerDefinitions {
            var layer : NeuralLayer
            switch (layerDefinition.layerType) {
            case .simpleFeedForwardWithNodes:
                layer = SimpleNeuralLayerWithNodes(numInputs: numInputsFromPreviousLayer, layerDefinition: layerDefinition)
            case .simpleFeedForward:
                layer = SimpleNeuralLayer(numInputs: numInputsFromPreviousLayer, layerDefinition: layerDefinition)
            case .simpleRecurrentWithNodes:
                layer = RecurrentNeuralLayerWithNodes(numInputs: numInputsFromPreviousLayer, layerDefinition: layerDefinition)
            case .simpleRecurrent:
                layer = RecurrentNeuralLayer(numInputs: numInputsFromPreviousLayer, layerDefinition: layerDefinition)
            case .lstm:
                layer = LSTMNeuralLayer(numInputs: numInputsFromPreviousLayer, layerDefinition: layerDefinition)
            }
            layers.append(layer)
            numInputsFromPreviousLayer = layerDefinition.numNodes
            if (layerDefinition.layerType == .simpleRecurrent) { hasRecurrentLayers = true }
            if (layerDefinition.layerType == .lstm) { hasRecurrentLayers = true }
        }
        
        numOutputs = layers.last!.getNodeCount()

    }
    
    open func getInputDimension() -> Int
    {
        return numInputs
    }
    open func getOutputDimension() -> Int
    {
        return numOutputs
    }
    open func getParameterDimension() -> Int
    {
        var count = 0
        for layer in layers {
            count += layer.getNodeCount() * layer.getWeightsPerNode()
        }
        return count
    }
    open func getNumberOfClasses() -> Int
    {
        if (numOutputs == 1) { return 2 }
        return numOutputs
    }
    
    open func setNeuralWeightUpdateMethod(_ method: NeuralWeightUpdateMethod, _ parameter: Double?)
    {
        for layer in layers {
            layer.setNeuralWeightUpdateMethod(method, parameter)
        }
    }
    
    ///  FeedForward routine
    open func feedForward(_ inputs: [Double]) -> [Double] {
        var x = inputs
        
        //  Go through each layer
        for layer in layers {
            //  Add a bias constant 1.0 to the input array
            x.append(1.0)
            
            //  Calculate the outputs from the layer
            x = layer.feedForward(x)
        }
        
        return x
    }
    
    open func setParameters(_ parameters: [Double]) throws
    {
        if (parameters.count < getParameterDimension()) { throw MachineLearningError.notEnoughData }
        
        //  If enough for all parameters, split and send to layers
        if (getParameterDimension() >= parameters.count) {
            var startIndex = 0
            for layer in layers {
                let numWeightsForLayer = layer.getNodeCount() * layer.getWeightsPerNode()
                let subArray = Array(parameters[startIndex...(startIndex+numWeightsForLayer-1)])
                layer.initWeights(subArray)
                startIndex += numWeightsForLayer
            }
        }
        else {
            //  Otherwise, send the same set to each layer
            for layer in layers {
                layer.initWeights(parameters)
            }
        }
    }

    
    ///  Method to set a custom function to initialize the parameters.  If not set, random parameters are used
    open func setCustomInitializer(_ function: ((_ trainData: MLDataSet)->[Double])!)
    {
        initializeFunction = function
    }
    
    open func getParameters() throws -> [Double]
    {
        var parameters : [Double] = []
        for layer in layers {
            parameters += layer.getWeights()
        }
        return parameters
    }
    
    ///  Method to initialize the weights - call before any training other than 'trainClassifier' or 'trainRegressor', which call this
    open func initializeWeights(_ trainData: MLDataSet!)
    {
        if let initFunc = initializeFunction, let data = trainData {
            let startWeights = initFunc(data)
            //  If enough for all parameters, split and send to layers
            if (getParameterDimension() == startWeights.count) {
                var startIndex = 0
                for layer in layers {
                    let numWeightsForLayer = layer.getNodeCount() * layer.getWeightsPerNode()
                    let subArray = Array(startWeights[startIndex...(startIndex+numWeightsForLayer-1)])
                    layer.initWeights(subArray)
                    startIndex += numWeightsForLayer
                }
            }
            else {
                //  Otherwise, send the same set to each layer
                for layer in layers {
                    layer.initWeights(startWeights)
                }
            }
        }
        else {
            //  No initialization function, set weights to random values
            for layer in layers {
                layer.initWeights(nil)
            }
        }
    }
    
    open func trainClassifier(_ trainData: MLClassificationDataSet) throws
    {
        initializeWeights(trainData)
        
        let epochCount = trainData.size * 2
        var epochSize = trainData.size
        if (trainData.size > 50) {
            epochSize = trainData.size / 10
        }
        else if (trainData.size > 10) {
            epochSize = trainData.size / 2
        }
        try classificationSGDBatchTrain(trainData, epochSize: epochSize, epochCount : epochCount, trainingRate: trainingRate, weightDecay: weightDecay)
    }
    
    open func continueTrainingClassifier(_ trainData: MLClassificationDataSet) throws
    {

        let epochCount = trainData.size * 2
        var epochSize = trainData.size
        if (trainData.size > 50) {
            epochSize = trainData.size / 10
        }
        else if (trainData.size > 10) {
            epochSize = trainData.size / 2
        }
        try classificationSGDBatchTrain(trainData, epochSize: epochSize, epochCount : epochCount, trainingRate: trainingRate, weightDecay: weightDecay)
    }
    
    
    ///  Return a 0 based index of the best neuron output (giving the most probable class for the input)
    open func classifyOne(_ inputs: [Double]) -> Int {
        //  Get the network outputs
        lastClassificationOutput = feedForward(inputs)
        
        if let outputs = lastClassificationOutput {
            //  If only one output, get the decision value and check
            if (outputs.count == 1) {
                let limit = layers.last!.getSingleNodeClassifyValue()
                if (outputs[0] > limit) { return 1 }
                return 0
            }
            
            //  Find the largest result
            var bestResult = -Double.infinity
            var bestNeuron = 0
            for index in 0..<outputs.count {
                if (outputs[index] > bestResult) {
                    bestResult = outputs[index]
                    bestNeuron = index
                }
            }
        
            return bestNeuron
        }
        return 0
    }
    
    ///  Set the 0 based index of the best neuron output (giving the most probable class for the input), for each point in a data set
    open func classify(_ testData: MLClassificationDataSet) throws {
        //  Verify the data set is the right type
        if (testData.dataType != .classification) { throw DataTypeError.invalidDataType }
        if (testData.inputDimension != numInputs) { throw DataTypeError.wrongDimensionOnInput }
        
        //  Classify each input
        for point in 0..<testData.size {
            do {
                let inputs = try testData.getInput(point)
                try testData.setClass(point, newClass: classifyOne(inputs))
            }
            catch { print("error indexing test data array") }
        }
    }
    
    open func trainRegressor(_ trainData: MLRegressionDataSet) throws
    {
        initializeWeights(trainData)
        
        let epochCount = trainData.size * 2
        let epochSize = trainData.size / 10
        SGDBatchTrain(trainData, epochSize: epochSize, epochCount : epochCount, trainingRate: trainingRate, weightDecay: weightDecay)
    }
    
    open func continueTrainingRegressor(_ trainData: MLRegressionDataSet) throws
    {
        let epochCount = trainData.size * 2
        let epochSize = trainData.size / 10
        SGDBatchTrain(trainData, epochSize: epochSize, epochCount : epochCount, trainingRate: trainingRate, weightDecay: weightDecay)
    }
    
    open func predictOne(_ inputs: [Double]) ->[Double]
    {
        return feedForward(inputs)
    }
    
    open func predict(_ testData: MLRegressionDataSet) throws
    {
        //  Verify the data set is the right type
        if (testData.dataType != .regression) { throw DataTypeError.invalidDataType }
        if (testData.inputDimension != numInputs) { throw DataTypeError.wrongDimensionOnInput }
        if (testData.outputDimension != layers.last?.getNodeCount()) { throw DataTypeError.wrongDimensionOnOutput }
        
        //  predict on each input
        for point in 0..<testData.size {
            do {
                let inputs = try testData.getInput(point)
                try testData.setOutput(point, newOutput: predictOne(inputs))
            }
            catch { print("error indexing test data array") }
        }
    }
    
    
    ///  Train on a sequence of data.  Be sure to initialize the weights before using the first time
    open func predictSequence(_ sequence: MLRegressionDataSet) throws
    {
        //  Start the sequence
        for layer in layers {
            layer.resetSequence()
        }
        
        //  Feed each item to the network
        try predict(sequence)
    }
    
    ///  Update weight change accumulations with a given gradient (𝟃E/𝟃h)
    open func trainWithGradient(_ inputs: [Double], gradient 𝟃E𝟃h : [Double])
    {
        //  Calculate the 𝟃E𝟃zs for the final layer
        layers.last!.getFinalLayer𝟃E𝟃zs(𝟃E𝟃h)
        
        //  Get the 𝟃E𝟃zs for the other layers
        if (layers.count > 1) {
            for nLayerIndex in stride(from: (layers.count - 2), through: 0, by: -1)
            {
                layers[nLayerIndex].getLayer𝟃E𝟃zs(layers[nLayerIndex+1])
            }
        }
        
        //  Set the inputs for calculating the weight changes
        var x = inputs
        
        //  Go through each layer, getting the weight changes
        for layer in layers {
            //  Add a bias constant 1.0 to the input array
            x.append(1.0)
            
            //  Update the weight changes
            x = layer.appendWeightChanges(x)
        }
    }

    ///  Train on one data item.  Be sure to initialize the weights before using the first time
    open func trainOne(_ inputs: [Double], expectedOutputs: [Double], trainingRate: Double, weightDecay: Double)
    {
        //  Get the results of a feedForward run (each node/layer remembers its own output)
        let h = feedForward(inputs)
        
        //  Calculate 𝟃E/𝟃h - the error with respect to the outputs
        //  For now, we are hard-coding a least squared error  E = 0.5 * (h - expected)²  -->   𝟃E/𝟃h = (h - expected)
        var 𝟃E𝟃h = [Double](repeating: 0.0, count: numOutputs)
        vDSP_vsubD(expectedOutputs, 1, h, 1, &𝟃E𝟃h, 1, vDSP_Length(numOutputs))
        
        //  Calculate the 𝟃E𝟃zs for the final layer
        layers.last!.getFinalLayer𝟃E𝟃zs(𝟃E𝟃h)
        
        //  Get the 𝟃E𝟃zs for the other layers
        if (layers.count > 1) {
            for nLayerIndex in stride(from: (layers.count - 2), through: 0, by: -1)
            {
                layers[nLayerIndex].getLayer𝟃E𝟃zs(layers[nLayerIndex+1])
            }
        }
        
        //  Set the inputs for calculating the weight changes
        var x = inputs
        
        //  Go through each layer, getting the weight changes
        for layer in layers {
            //  Add a bias constant 1.0 to the input array
            x.append(1.0)
            
            //  Update the weights
            layer.clearWeightChanges()
            x = layer.appendWeightChanges(x)
            layer.updateWeightsFromAccumulations(trainingRate, weightDecay: weightDecay)
        }
    }
    
    ///  Train on a sequence of data.  Be sure to initialize the weights before using the first time
    open func trainSequence(_ sequence: MLRegressionDataSet, trainingRate: Double, weightDecay: Double)
    {
        //  Start the sequence
        //  Clear the weight change accumulations
        for layer in layers {
            layer.resetSequence()
            layer.clearWeightChanges()
        }
        
        //  Train on each element of the sequence, as if it was a batch
        
        //  Iterate through each training datum in the batch, feeding forward, but remembering the recurrent values
        for dataIndex in 0..<sequence.size {
            //  Get the results of a feedForward run (each node remembers its own output)
            do {
                let inputs = try sequence.getInput(dataIndex)
                _ = feedForward(inputs)
            }
            catch { print("error indexing sequence data array") }
            for layer in layers {
                layer.storeRecurrentValues()
            }
        }
        
        //  Iterate backwards through each training datum in the batch, backpropogating
        for dataIndex in stride(from: (sequence.size - 1), through: 0, by: -1) {
            do {
                //  Back the output value for all layers off the stack
                for layer in layers {
                    layer.retrieveRecurrentValues(dataIndex)
                }
                
                //  Get the last outputs from the last layer
                let h = layers.last!.getLastOutput()
                
                //  Calculate 𝟃E/𝟃h - the error with respect to the outputs
                //  For now, we are hard-coding a least squared error  E = 0.5 * (h - expected)²  -->   𝟃E/𝟃h = (h - expected)
                let expectedOutputs = try sequence.getOutput(dataIndex)
                
                var 𝟃E𝟃h = [Double](repeating: 0.0, count: numOutputs)
                vDSP_vsubD(expectedOutputs, 1, h, 1, &𝟃E𝟃h, 1, vDSP_Length(numOutputs))
                
                //  Calculate the 𝟃E𝟃z for the final layer
                layers.last!.getFinalLayer𝟃E𝟃zs(𝟃E𝟃h)
                
                //  Get the 𝟃E𝟃zs for the other layers
                if (layers.count > 1) {
                    for nLayerIndex in stride(from: (layers.count - 2), through: 0, by: -1)
                    {
                        layers[nLayerIndex].getLayer𝟃E𝟃zs(layers[nLayerIndex+1])
                    }
                }
            
                //  Set the inputs for calculating the weight changes
                var x = try sequence.getInput(dataIndex)
                
                //  Go through each layer
                for layer in layers {
                    //  Add a bias constant 1.0 to the input array
                    x.append(1.0)
                    
                    //  Append the weight changes for the layer
                    x = layer.appendWeightChanges(x)
                }
            }
            catch { print("error indexing sequence data array") }
        }
        
        //  Update the weights based on the weight change accumulations
        updateWeights(trainingRate: trainingRate, weightDecay: weightDecay)
    }
    
    ///  Train on a batch data item.  Be sure to initialize the weights before using the first time
    open func batchTrain(_ trainData: MLRegressionDataSet, epochIndices : [Int], trainingRate: Double, weightDecay: Double)
    {
        //  Clear the weight change accumulations
        clearWeightChanges()
        
        //  Iterate through each training datum in the batch
        for dataIndex in 0..<epochIndices.count {
            //  Get the results of a feedForward run (each node remembers its own output)
            do {
                var x = try trainData.getInput(epochIndices[dataIndex])
                _ = feedForward(x)
                
                //  Get the last outputs from the last layer
                let h = layers.last!.getLastOutput()
            
                //  Calculate 𝟃E/𝟃h - the error with respect to the outputs
                //  For now, we are hard-coding a least squared error  E = 0.5 * (h - expected)²  -->   𝟃E/𝟃h = (h - expected)
                let expectedOutput = try trainData.getOutput(epochIndices[dataIndex])
                
                var 𝟃E𝟃h = [Double](repeating: 0.0, count: numOutputs)
                vDSP_vsubD(expectedOutput, 1, h, 1, &𝟃E𝟃h, 1, vDSP_Length(numOutputs))
                
                //  Calculate the 𝟃E𝟃z for the final layer
                layers.last!.getFinalLayer𝟃E𝟃zs(𝟃E𝟃h)
                
                //  Get the 𝟃E𝟃zs for the other layers
                if (layers.count > 1) {
                    for nLayerIndex in stride(from: (layers.count - 2), through: 0, by: -1)
                    {
                        layers[nLayerIndex].getLayer𝟃E𝟃zs(layers[nLayerIndex+1])
                    }
                }
                
                //  Go through each layer
                for layer in layers {
                    //  Add a bias constant 1.0 to the input array
                    x.append(1.0)
                    
                    //  Append the weight changes for the layer
                    x = layer.appendWeightChanges(x)
                }
            }
            catch { print("error indexing sequence data array") }
        }
        
        //  Update the weights based on the weight change accumulations
        let averageTrainingRate = trainingRate / Double(epochIndices.count)  //  Make negative so we can use DSP to vectorize equation
        updateWeights(trainingRate: averageTrainingRate, weightDecay: weightDecay)
    }
    
    ///  Zero out all the weight changes
    open func clearWeightChanges()
    {
        for layer in layers {
            layer.clearWeightChanges()
        }
    }
    
    ///  Update the weights from the accumulated changes
    open func updateWeights(trainingRate: Double, weightDecay: Double)
    {
        for layer in layers {
            layer.updateWeightsFromAccumulations(trainingRate, weightDecay: weightDecay)
        }
    }

    ///  Train a network on a set of data.  Be sure to initialize the weights before using the first time
    open func SGDBatchTrain(_ trainData: MLRegressionDataSet, epochSize: Int, epochCount : Int, trainingRate: Double, weightDecay: Double)
    {
        //  Create the batch indices array
        var batchIndices = [Int](repeating: 0, count: epochSize)
        
        //  Run each epoch
        for _ in 0..<epochCount {
            //  Get training set indices for this epoch
            for index in 0..<epochSize {
#if os(Linux)
                    batchIndices[index] = Int(random() % trainData.size)
#else
                    batchIndices[index] = Int(arc4random_uniform(UInt32(trainData.size)))
#endif
            }
            
            //  Learn on this epoch
            batchTrain(trainData, epochIndices: batchIndices, trainingRate: trainingRate, weightDecay: weightDecay)
        }
    }
    
    ///  Train a classification network for a single instance.  Be sure to initialize the weights before using the first time
    open func classificationTrainOne(_ inputs: [Double], expectedOutput: Int, trainingRate: Double, weightDecay: Double)
    {
        //  Get the false level for the output layer
        var falseLevel = 0.0
        if (layers.last!.getActivation() == .hyperbolicTangent) {falseLevel = -1.0}
        
        //  Create the expected output array for expected class
        var expectedOutputs : [Double] = []
        if (numOutputs > 1) {
            for classIndex in 0..<numOutputs {
                expectedOutputs.append(classIndex == expectedOutput ? 1.0 : falseLevel)
            }
        }
        else {
            expectedOutputs.append(expectedOutput == 1 ? 1.0 : falseLevel)
        }
        
        //  Do the normal training
        trainOne(inputs, expectedOutputs: expectedOutputs, trainingRate: trainingRate, weightDecay: weightDecay)
    }
    
    ///  Get the gradient from the last classification output and the result taken (used for policy gradient reinforcement learning)
    open func getLastClassificationGradient(resultUsed : Int) -> [Double]
    {
        //  Get the false level for the output layer
        var falseLevel = 0.0
        if (layers.last!.getActivation() == .hyperbolicTangent) {falseLevel = -1.0}
        
        //  Create the expected output array for the used class
        var expectedOutputs : [Double] = []
        if (numOutputs > 1) {
            for classIndex in 0..<numOutputs {
                expectedOutputs.append(classIndex == resultUsed ? 1.0 : falseLevel)
            }
        }
        else {
            expectedOutputs.append(resultUsed == 1 ? 1.0 : falseLevel)
        }
        
        //  Get the gradient
        var gradient = [Double](repeating: 0.0, count: numOutputs)
        if (lastClassificationOutput == nil) { return gradient }
        vDSP_vsubD(expectedOutputs, 1, lastClassificationOutput!, 1, &gradient, 1, vDSP_Length(numOutputs))     // grad = last - expected
        
        return gradient
    }
    
    ///  Train a classification network on a set of data.  Be sure to initialize the weights before using the first time
    open func classificationSGDBatchTrain(_ trainData: MLClassificationDataSet, epochSize: Int, epochCount : Int, trainingRate: Double, weightDecay: Double) throws
    {
        //  Verify the data set is the right type
        if (trainData.dataType != .classification) { throw DataTypeError.invalidDataType }
        
        //  Get the false level for the output layer
        var falseLevel = 0.0
        if (layers.last!.getActivation() == .hyperbolicTangent) {falseLevel = -1.0}
        
        if trainData.dataType == .realAndClass {
            let combinedData = trainData as! MLCombinedDataSet
            //  Create the expected output array for expected class
            do {
                var sampleIndex = 0
                for index in 0..<trainData.size {
                    let classValue = try trainData.getClass(index)
                    var outputs  : [Double] = []
                    if (numOutputs > 1) {
                        for classIndex in 0..<numOutputs {
                            outputs.append(classIndex == classValue ? 1.0 : falseLevel)
                        }
                    }
                    else {
                        outputs.append(classValue == 1 ? 1.0 : falseLevel)
                    }
                    try combinedData.setOutput(sampleIndex, newOutput: outputs)
                    sampleIndex += 1
                }
            }
            
            //  Train using the normal routine
            SGDBatchTrain(combinedData, epochSize : epochSize, epochCount : epochCount, trainingRate : trainingRate, weightDecay: weightDecay)
        }
        else {
            //  We have a classification data set.  We need to convert it to a regression data set
            if let regressionData = DataSet(dataType: .regression, withInputsFrom: trainData) {
                //  Create the expected output array for expected class
                do {
                    var sampleIndex = 0
                    for index in 0..<trainData.size {
                        let classValue = try trainData.getClass(index)
                        var outputs  : [Double] = []
                        if (numOutputs > 1) {
                            for classIndex in 0..<numOutputs {
                                outputs.append(classIndex == classValue ? 1.0 : falseLevel)
                            }
                        }
                        else {
                            outputs.append(classValue == 1 ? 1.0 : falseLevel)
                        }
                        try regressionData.setOutput(sampleIndex, newOutput: outputs)
                        sampleIndex += 1
                    }
                }
                
                //  Train using the normal routine
                SGDBatchTrain(regressionData, epochSize : epochSize, epochCount : epochCount, trainingRate : trainingRate, weightDecay: weightDecay)
            }
            else {
                throw MachineLearningError.dataWrongDimension       //  Only error that can come out of DataSet creation
            }
        }
    }
    
    ///  Decay weights for regularization.  All weights are multiplied by the constant supplied as the parameter
    ///  The parameter is eta * lamba / n --> learning rate * regularization term divided by sample size
    ///  The weights for the bias term are skipped
    open func decayWeights(_ decayFactor : Double)
    {
        for layer in layers {
            layer.decayWeights(decayFactor)
        }
    }
    
    ///  Function to perform a gradient check on the network
    public func gradientCheck(inputs: [Double], expectedOutputs: [Double], ε: Double, Δ: Double) -> Bool
    {
        //  Feed forward and do a single gradient update
        //  Get the results of a feedForward run (each node remembers its own output)
        var h = feedForward(inputs)
        if (hasRecurrentLayers) {
            //  Recurrent layers need their outputs saved for the recurrent inputs
            for layer in layers {
                layer.storeRecurrentValues()
            }
            //  Calculate h using the stored inputs
            h = feedForward(inputs)
        }
        
        //  Calculate 𝟃E/𝟃h - the error with respect to the outputs
        //  For now, we are hard-coding a least squared error  E = 0.5 * (h - expected)²  -->   𝟃E/𝟃h = (h - expected)
        var 𝟃E𝟃h = [Double](repeating: 0.0, count: numOutputs)
        vDSP_vsubD(expectedOutputs, 1, h, 1, &𝟃E𝟃h, 1, vDSP_Length(numOutputs))
        
        //  Calculate the 𝟃E𝟃zs for the final layer
        layers.last!.getFinalLayer𝟃E𝟃zs(𝟃E𝟃h)
        
        //  Get the 𝟃E𝟃zs for the other layers
        if (layers.count > 1) {
            for nLayerIndex in stride(from: (layers.count - 2), through: 0, by: -1)
            {
                layers[nLayerIndex].getLayer𝟃E𝟃zs(layers[nLayerIndex+1])
            }
        }
        
        //  Set the inputs for calculating the weight changes
        var x = inputs
        
        //  Go through each layer, getting the weight changes
        for layer in layers {
            //  Add a bias constant 1.0 to the input array
            x.append(1.0)
            
            //  Calculate the outputs from the layer
            layer.clearWeightChanges()
            x = layer.appendWeightChanges(x)
        }
       
        //  Have each layer check their gradients
        expectedOutput = expectedOutputs
        var result = true
        for layer in layers {
            if (!layer.gradientCheck(x: inputs, ε: ε, Δ: Δ, network: self)) { result = false }
        }
        return result
    }
    
    public func getResultLoss() throws -> [Double]
    {
        if (expectedOutput == nil) { throw NeuralNetworkError.expectedOutputNotSet }
        
        let finalResults = layers.last!.getLastOutput()
        var loss = [Double](repeating: 0.0, count: finalResults.count)
        
        //  Get the error vector
        var errorVector = [Double](repeating: 0.0, count: finalResults.count)
        vDSP_vsubD(expectedOutput!, 1, finalResults, 1, &errorVector, 1, vDSP_Length(finalResults.count))
        
        //  Assume squared error loss
        vDSP_vsqD(errorVector, 1, &loss, 1, vDSP_Length(finalResults.count))
        var half = 0.5
        vDSP_vsmulD(loss, 1, &half, &loss, 1, vDSP_Length(finalResults.count))
        
        return loss
    }
}
