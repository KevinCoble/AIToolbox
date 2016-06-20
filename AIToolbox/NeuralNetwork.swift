//
//  NeuralNetwork.swift
//  AIToolbox
//
//  Created by Kevin Coble on 10/6/15.
//  Copyright © 2015 Kevin Coble. All rights reserved.
//

import Foundation
import Accelerate

public enum NeuronLayerType {
    case SimpleFeedForward
    case SimpleRecurrent
    case LSTM
}

public enum NeuralActivationFunction {
    case None
    case HyperbolicTangent
    case Sigmoid
    case SigmoidWithCrossEntropy
    case RectifiedLinear
    case SoftSign
    case SoftMax        //  Only valid on output (last) layer
}

protocol NeuralLayer {
    func getNodeCount() -> Int
    func getWeightsPerNode()-> Int
    func getActivation()-> NeuralActivationFunction
    func feedForward(x: [Double]) -> [Double]
    func initWeights(startWeights: [Double]!)
    func getWeights() -> [Double]
    func getLastOutput() -> [Double]
    func getFinalLayer𝟃E𝟃zs(𝟃E𝟃h: [Double])
    func getLayer𝟃E𝟃zs(nextLayer: NeuralLayer)
    func get𝟃E𝟃hForNodeInPreviousLayer(inputIndex: Int) ->Double
    func clearWeightChanges()
    func appendWeightChanges(inputs: [Double]) -> [Double]
    func updateWeightsFromAccumulations(averageTrainingRate: Double, weightDecay: Double)
    func decayWeights(decayFactor : Double)
    func getSingleNodeClassifyValue() -> Double
    func resetSequence()
    func storeRecurrentValues()
    func retrieveRecurrentValues(sequenceIndex: Int)
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
    func initWeights(startWeights: [Double]!)
    {
        if let startWeights = startWeights {
            if (startWeights.count == 1) {
                W = [Double](count: numWeights, repeatedValue: startWeights[0])
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
    }
    
    func feedForward(x: [Double]) -> Double
    {
        //  Get the weighted sum:  z = W⋅x
        var z = 0.0
        vDSP_dotprD(W, 1, x, 1, &z, vDSP_Length(W.count))
        
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
    
    func get𝟃E𝟃z()
    {
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
                //  Should not get here - SoftMax is only valid on output layer
                break
        }
    }
    
    func clearWeightChanges()
    {
        𝟃E𝟃W = [Double](count: W.count, repeatedValue: 0.0)
    }
    
    func appendWeightChanges(x: [Double]) -> Double
    {
        //  Update each weight accumulation
        //  𝟃E/𝟃W = 𝟃E/𝟃z ⋅ 𝟃z/𝟃W = 𝟃E/𝟃z ⋅ x
        vDSP_vsmaD(x, 1, &𝟃E𝟃z, 𝟃E𝟃W, 1, &𝟃E𝟃W, 1, vDSP_Length(numWeights))
        
        return h    //  return output for next layer
    }
    
    func updateWeightsFromAccumulations(averageTrainingRate: Double)
    {
        //  Update the weights from the accumulations
        //  W -= 𝟃E𝟃W, * averageTrainingRate
        var η = -averageTrainingRate     //  Needed for unsafe pointer conversion - negate for multiply-and-add vector operation
        vDSP_vsmaD(𝟃E𝟃W, 1, &η, W, 1, &W, 1, vDSP_Length(numWeights))
    }
    
    func decayWeights(decayFactor : Double)
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
}

final class SimpleNeuralLayer: NeuralLayer {
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
        var outputs : [Double] = []
        //  Assume input array already has bias constant 1.0 appended
        //  Fully-connected nodes means all nodes get the same input array
        if (nodes[0].activation == .SoftMax) {
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
    
    func getFinalLayer𝟃E𝟃zs(𝟃E𝟃h: [Double])
    {
        for nodeIndex in 0..<nodes.count {
            //  Set 𝟃E/𝟃h from external error
            nodes[nodeIndex].𝟃E𝟃h = 𝟃E𝟃h[nodeIndex]
            
            //  Backpropogate error to the z value level
            nodes[nodeIndex].get𝟃E𝟃z()
        }
    }
    
    func getLayer𝟃E𝟃zs(nextLayer: NeuralLayer)
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
        var outputs : [Double] = []
        //  Assume input array already has bias constant 1.0 appended
        //  Fully-connected nodes means all nodes get the same input array
        for node in nodes {
            outputs.append(node.appendWeightChanges(x))
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
        //  Set the last recurrent value in the history array to the last output
        for node in nodes {
            node.getLastRecurrentValue()
        }
    }
}

public class NeuralNetwork: Classifier, Regressor {
    //  Layers
    let numInputs : Int
    let numOutputs : Int
    var layers : [NeuralLayer]
    var trainingRate = 0.3
    var weightDecay = 1.0
    var initializeFunction : ((trainData: DataSet)->[Double])!
    var hasRecurrentLayers = false
    
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
            case .SimpleFeedForward:
                layer = SimpleNeuralLayer(numInputs: numInputsFromPreviousLayer, layerDefinition: layerDefinition)
            case .SimpleRecurrent:
                layer = RecurrentNeuralLayer(numInputs: numInputsFromPreviousLayer, layerDefinition: layerDefinition)
            case .LSTM:
                layer = LSTMNeuralLayer(numInputs: numInputsFromPreviousLayer, layerDefinition: layerDefinition)
            }
            layers.append(layer)
            numInputsFromPreviousLayer = layerDefinition.numNodes
            if (layerDefinition.layerType == .SimpleRecurrent) { hasRecurrentLayers = true }
            if (layerDefinition.layerType == .LSTM) { hasRecurrentLayers = true }
        }
        
        numOutputs = layers.last!.getNodeCount()

    }
    
    public func getInputDimension() -> Int
    {
        return numInputs
    }
    public func getOutputDimension() -> Int
    {
        return numOutputs
    }
    public func getParameterDimension() -> Int
    {
        var count = 0
        for layer in layers {
            count += layer.getNodeCount() * layer.getWeightsPerNode()
        }
        return count
    }
    public func getNumberOfClasses() -> Int
    {
        let numberOfOutputNodes = layers.last!.getNodeCount()
        if (numberOfOutputNodes == 1) { return 2 }
        return numberOfOutputNodes
    }
    
    ///  FeedForward routine
    public func feedForward(inputs: [Double]) -> [Double] {
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
    
    public func setParameters(parameters: [Double]) throws
    {
        if (parameters.count < getParameterDimension()) { throw MachineLearningError.NotEnoughData }
        
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
    public func setCustomInitializer(function: ((trainData: DataSet)->[Double])!)
    {
        initializeFunction = function
    }
    
    public func getParameters() throws -> [Double]
    {
        var parameters : [Double] = []
        for layer in layers {
            parameters += layer.getWeights()
        }
        return parameters
    }
    
    ///  Method to initialize the weights - call before any training other than 'trainClassifier' or 'trainRegressor', which call this
    public func initializeWeights(trainData: DataSet!)
    {
        if let initFunc = initializeFunction, data = trainData {
            let startWeights = initFunc(trainData: data)
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
    
    public func trainClassifier(trainData: DataSet) throws
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
    
    public func continueTrainingClassifier(trainData: DataSet) throws
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
    public func classifyOne(inputs: [Double]) -> Int {
        //  Get the network outputs
        let outputs = feedForward(inputs)
        
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
    
    ///  Set the 0 based index of the best neuron output (giving the most probable class for the input), for each point in a data set
    public func classify(testData: DataSet) throws {
        //  Verify the data set is the right type
        if (testData.dataType != .Classification) { throw DataTypeError.InvalidDataType }
        if (testData.inputDimension != numInputs) { throw DataTypeError.WrongDimensionOnInput }
        
        //  Classify each input
        testData.classes = []
        for point in 0..<testData.size {
            do {
                let inputs = try testData.getInput(point)
                try testData.setClass(point, newClass: classifyOne(inputs))
            }
            catch { print("error indexing test data array") }
        }
    }
    
    public func trainRegressor(trainData: DataSet) throws
    {
        initializeWeights(trainData)
        
        let epochCount = trainData.size * 2
        let epochSize = trainData.size / 10
        SGDBatchTrain(trainData, epochSize: epochSize, epochCount : epochCount, trainingRate: trainingRate, weightDecay: weightDecay)
    }
    
    public func continueTrainingRegressor(trainData: DataSet) throws
    {
        let epochCount = trainData.size * 2
        let epochSize = trainData.size / 10
        SGDBatchTrain(trainData, epochSize: epochSize, epochCount : epochCount, trainingRate: trainingRate, weightDecay: weightDecay)
    }
    
    public func predictOne(inputs: [Double]) ->[Double]
    {
        return feedForward(inputs)
    }
    
    public func predict(testData: DataSet) throws
    {
        //  Verify the data set is the right type
        if (testData.dataType != .Regression) { throw DataTypeError.InvalidDataType }
        if (testData.inputDimension != numInputs) { throw DataTypeError.WrongDimensionOnInput }
        if (testData.outputDimension != layers.last?.getNodeCount()) { throw DataTypeError.WrongDimensionOnOutput }
        
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
    public func predictSequence(sequence: DataSet) throws
    {
        //  Start the sequence
        for layer in layers {
            layer.resetSequence()
        }
        
        //  Feed each item to the network
        try predict(sequence)
    }

    ///  Train on one data item.  Be sure to initialize the weights before using the first time
    public func trainOne(inputs: [Double], expectedOutputs: [Double], trainingRate: Double, weightDecay: Double)
    {
        //  Get the results of a feedForward run (each node remembers its own output)
        let h = feedForward(inputs)
        
        //  Calculate 𝟃E/𝟃h - the error with respect to the outputs
        //  For now, we are hard-coding a least squared error  E = 0.5 * (h - expected)²  -->   𝟃E/𝟃h = (h - expected)
        var 𝟃E𝟃h = [Double](count:numOutputs, repeatedValue: 0.0)
        vDSP_vsubD(expectedOutputs, 1, h, 1, &𝟃E𝟃h, 1, vDSP_Length(numOutputs))
        
        //  Calculate the 𝟃E𝟃zs for the final layer
        layers.last!.getFinalLayer𝟃E𝟃zs(𝟃E𝟃h)
        
        //  Get the 𝟃E𝟃zs for the other layers
        if (layers.count > 1) {
            for nLayerIndex in (layers.count - 2).stride(through: 0, by: -1)
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
            layer.updateWeightsFromAccumulations(trainingRate, weightDecay: weightDecay)
        }
    }
    
    ///  Train on a sequence of data.  Be sure to initialize the weights before using the first time
    public func trainSequence(sequence: DataSet, trainingRate: Double, weightDecay: Double)
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
                feedForward(inputs)
            }
            catch { print("error indexing sequence data array") }
            for layer in layers {
                layer.storeRecurrentValues()
            }
        }
        
        //  Iterate backwards through each training datum in the batch, backpropogating
        for dataIndex in (sequence.size - 1).stride(through: 0, by: -1) {
            do {
                //  Back the output value for all layers off the stack
                for layer in layers {
                    layer.retrieveRecurrentValues(dataIndex)
                }
                
                //  Get the last outputs from the last layer
                let h = layers.last!.getLastOutput()
                
                //  Calculate 𝟃E/𝟃h - the error with respect to the outputs
                //  For now, we are hard-coding a least squared error  E = 0.5 * (h - expected)²  -->   𝟃E/𝟃h = (h - expected)
                let expectedOutput = try sequence.getOutput(dataIndex)
                
                var 𝟃E𝟃h = [Double](count:numOutputs, repeatedValue: 0.0)
                vDSP_vsubD(expectedOutput, 1, h, 1, &𝟃E𝟃h, 1, vDSP_Length(numOutputs))
                
                //  Calculate the 𝟃E𝟃z for the final layer
                layers.last!.getFinalLayer𝟃E𝟃zs(𝟃E𝟃h)
                
                //  Get the 𝟃E𝟃zs for the other layers
                if (layers.count > 1) {
                    for nLayerIndex in (layers.count - 2).stride(through: 0, by: -1)
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
        for layer in layers {
            layer.updateWeightsFromAccumulations(trainingRate, weightDecay: weightDecay)
        }
    }
    
    ///  Train on a batch data item.  Be sure to initialize the weights before using the first time
    public func batchTrain(trainData: DataSet, epochIndices : [Int], trainingRate: Double, weightDecay: Double)
    {
        //  Clear the weight change accumulations
        for layer in layers {
            layer.clearWeightChanges()
        }
        
        //  Iterate through each training datum in the batch
        for dataIndex in 0..<epochIndices.count {
            //  Get the results of a feedForward run (each node remembers its own output)
            do {
                var x = try trainData.getInput(epochIndices[dataIndex])
                feedForward(x)
                
                //  Get the last outputs from the last layer
                let h = layers.last!.getLastOutput()
            
                //  Calculate 𝟃E/𝟃h - the error with respect to the outputs
                //  For now, we are hard-coding a least squared error  E = 0.5 * (h - expected)²  -->   𝟃E/𝟃h = (h - expected)
                let expectedOutput = try trainData.getOutput(epochIndices[dataIndex])
                
                var 𝟃E𝟃h = [Double](count:numOutputs, repeatedValue: 0.0)
                vDSP_vsubD(expectedOutput, 1, h, 1, &𝟃E𝟃h, 1, vDSP_Length(numOutputs))
                
                //  Calculate the 𝟃E𝟃z for the final layer
                layers.last!.getFinalLayer𝟃E𝟃zs(𝟃E𝟃h)
                
                //  Get the 𝟃E𝟃zs for the other layers
                if (layers.count > 1) {
                    for nLayerIndex in (layers.count - 2).stride(through: 0, by: -1)
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
        let averageTrainingRate = trainingRate * -1.0 / Double(epochIndices.count)  //  Make negative so we can use DSP to vectorize equation
        for layer in layers {
            layer.updateWeightsFromAccumulations(averageTrainingRate, weightDecay: weightDecay)
        }
    }

    ///  Train a network on a set of data.  Be sure to initialize the weights before using the first time
    public func SGDBatchTrain(trainData: DataSet, epochSize: Int, epochCount : Int, trainingRate: Double, weightDecay: Double)
    {
        //  Create the batch indices array
        var batchIndices = [Int](count: epochSize, repeatedValue: 0)
        
        //  Run each epoch
        for _ in 0..<epochCount {
            //  Get training set indices for this epoch
            for index in 0..<epochSize {
                batchIndices[index] = Int(arc4random_uniform(UInt32(trainData.size)))
            }
            
            //  Learn on this epoch
            batchTrain(trainData, epochIndices: batchIndices, trainingRate: trainingRate, weightDecay: weightDecay)
        }
    }
    
    ///  Train a classification network for a single instance.  Be sure to initialize the weights before using the first time
    public func classificationTrainOne(inputs: [Double], expectedOutput: Int, trainingRate: Double, weightDecay: Double)
    {
        //  Get the false level for the output layer
        var falseLevel = 0.0
        if (layers.last!.getActivation() == .HyperbolicTangent) {falseLevel = -1.0}
        
        //  Create the expected output array for expected class
        var expectedOutputs : [Double] = []
        for classIndex in 0..<layers.last!.getNodeCount() {
            expectedOutputs.append(classIndex == expectedOutput ? 1.0 : falseLevel)
        }
        
        //  Do the normal training
        trainOne(inputs, expectedOutputs: expectedOutputs, trainingRate: trainingRate, weightDecay: weightDecay)
    }
    
    ///  Train a classification network on a set of data.  Be sure to initialize the weights before using the first time
    public func classificationSGDBatchTrain(trainData: DataSet, epochSize: Int, epochCount : Int, trainingRate: Double, weightDecay: Double) throws
    {
        //  Verify the data set is the right type
        if (trainData.dataType != .Classification) { throw DataTypeError.InvalidDataType }
        
        //  Get the false level for the output layer
        var falseLevel = 0.0
        if (layers.last!.getActivation() == .HyperbolicTangent) {falseLevel = -1.0}
        
        //  Get the number of nodes in the last layer
        let numOutputNodes = layers.last!.getNodeCount()
        
        //  Create the expected output array for expected class
        do {
            var sampleIndex = 0
            for classValue in trainData.classes! {
                var outputs  : [Double] = []
                if (numOutputNodes > 1) {
                    for classIndex in 0..<numOutputNodes {
                        outputs.append(classIndex == classValue ? 1.0 : falseLevel)
                    }
                }
                else {
                    outputs.append(classValue == 1 ? 1.0 : falseLevel)
                }
                try trainData.setOutput(sampleIndex, newOutput: outputs)
                sampleIndex += 1
            }
        }
        
        //  Train using the normal routine
        SGDBatchTrain(trainData, epochSize : epochSize, epochCount : epochCount, trainingRate : trainingRate, weightDecay: weightDecay)
    }
    
    ///  Decay weights for regularization.  All weights are multiplied by the constant supplied as the parameter
    ///  The parameter is eta * lamba / n --> learning rate * regularization term divided by sample size
    ///  The weights for the bias term are skipped
    public func decayWeights(decayFactor : Double)
    {
        for layer in layers {
            layer.decayWeights(decayFactor)
        }
    }
}
