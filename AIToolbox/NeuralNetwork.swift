//
//  NeuralNetwork.swift
//  AIToolbox
//
//  Created by Kevin Coble on 10/6/15.
//  Copyright © 2015 Kevin Coble. All rights reserved.
//

import Foundation
import Accelerate

public enum NeuralActivationFunction {
    case None
    case HyberbolicTangent
    case Sigmoid
    case SigmoidWithCrossEntropy
    case RectifiedLinear
    case SoftSign
    case SoftMax        //  Only valid on output (last) layer
}

final class NeuralNode {
    //  Activation function
    let activation : NeuralActivationFunction
    var weights : [Double]
    var lastWeightedSum : Double //  Last weights dot-producted with inputs - remembered for training purposes
    var lastOutput : Double //  Last result calculated
    var delta : Double      //  Difference in expected output to calculated result - weighted sum from all nodes this node outputs too
    var accumulatedWeightChanges : [Double]?
    
    ///  Create the neural network node with a set activation function
    init(numInputs : Int, activationFunction: NeuralActivationFunction)
    {
        activation = activationFunction
        weights = []
        for _ in 0..<(numInputs)  { //  Add one weight for the bias term
            weights.append(DataSet.gaussianRandom(0.0, standardDeviation: 1.0 / Double(numInputs)))    //  input weights - Initialize to a random number to break initial symmetry of the network, scaled to the inputs
        }
        weights.append(DataSet.gaussianRandom(0.0, standardDeviation:1.0))    //  Bias weight - Initialize to a  random number to break initial symmetry of the network
        lastWeightedSum = 0.0
        lastOutput = 0.0
        delta = 0.0
    }
    
    func getNodeOutput(inputs: [Double]) -> Double
    {
        //  Get the weighted sum
        vDSP_dotprD(weights, 1, inputs, 1, &lastWeightedSum, vDSP_Length(weights.count))
        
        //  Use the activation function function for the nonlinearity
        switch (activation) {
            case .None:
                lastOutput = lastWeightedSum
                break
            case .HyberbolicTangent:
                lastOutput = tanh(lastWeightedSum)
                break
            case .SigmoidWithCrossEntropy:
                fallthrough
            case .Sigmoid:
                lastOutput = 1.0 / (1.0 + exp(-lastWeightedSum))
                break
            case .RectifiedLinear:
                lastOutput = lastWeightedSum
                if (lastWeightedSum < 0) { lastOutput = 0.0 }
                break
            case .SoftSign:
                lastOutput = lastWeightedSum / (1.0 + abs(lastWeightedSum))
                break
            case .SoftMax:
                lastOutput = exp(lastWeightedSum)
                break
        }
        
        return lastOutput
    }
    
    //  Get the partial derivitive of the error with respect to the weighted sum
    func getFinalNodeDelta(expectedOutput: Double)
    {
        //  error = (result - expected value)^2  (squared error) - not the case for softmax or cross entropy
        //  derivitive of error = 2 * (result - expected value) * result'  (chain rule - result is a function of the sum through the non-linearity)
        //  derivitive of the non-linearity: tanh' -> 1 - result^2, sigmoid -> result - result^2, rectlinear -> 0 if result<0 else 1
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
    
    func resetDelta()
    {
        delta = 0.0
    }
    
    func addToDelta(addition: Double)
    {
        delta += addition
    }
    
    func getWeightTimesDelta(weightIndex: Int) ->Double
    {
        return weights[weightIndex] * delta
    }
    
    func multiplyDeltaByNonLinearityDerivitive()
    {
        //  derivitive of the non-linearity: tanh' -> 1 - result^2, sigmoid -> result - result^2, rectlinear -> 0 if result<0 else 1
        switch (activation) {
            case .None:
                break
            case .HyberbolicTangent:
                delta *= (1 - lastOutput * lastOutput)
                break
            case .SigmoidWithCrossEntropy:
                fallthrough
            case .Sigmoid:
                delta *= (lastOutput - lastOutput * lastOutput)
                break
            case .RectifiedLinear:
                delta = lastOutput < 0.0 ? 0.0 : delta
                break
            case .SoftSign:
                if (lastOutput < 0) { delta *= -1 }
                delta /= (1.0 + lastOutput) * (1.0 + lastOutput)
                break
            case .SoftMax:
                //  Should not get here - SoftMax is only valid on output layer
                break
        }
    }
    
    func updateWeights(inputs: [Double], trainingRate: Double) -> Double
    {
        //  Update each weight
        var negativeScaleFactor = trainingRate * delta * -1.0  //  Make negative so we can use DSP to vectorize equation
        vDSP_vsmaD(inputs, 1, &negativeScaleFactor, weights, 1, &weights, 1, vDSP_Length(weights.count))    //  weights = weights + delta * inputs

        return lastOutput
    }
    
    func clearWeightChanges()
    {
        accumulatedWeightChanges = [Double](count: weights.count, repeatedValue: 0.0)
    }
    
    func appendWeightChanges(inputs: [Double]) -> Double
    {
        //  Update each weight accumulation
        vDSP_vsmaD(inputs, 1, &delta, accumulatedWeightChanges!, 1, &accumulatedWeightChanges!, 1, vDSP_Length(weights.count))       //  Calculate the weight change:  total change = total change + delta * inputs
        
        return lastOutput
    }
    
    func updateWeightsFromAccumulations(averageTrainingRate: Double)
    {
        //  Update the weights from the accumulations
        //  weights -= accumulation * averageTrainingRate  (training rate is passed in negative to allow subtraction with vector math)
        var x = averageTrainingRate     //  Needed for unsafe pointer conversion
        vDSP_vsmaD(accumulatedWeightChanges!, 1, &x, weights, 1, &weights, 1, vDSP_Length(weights.count))
    }
    
    func decayWeights(decayFactor : Double)
    {
        var x = decayFactor     //  Needed for unsafe pointer conversion
        vDSP_vsmulD(weights, 1, &x, &weights, 1, vDSP_Length(weights.count-1))
    }
}

final class NeuralLayer {
    //  Nodes
    var nodes : [NeuralNode]
    
    ///  Create the neural network layer based on a tuple (number of nodes, activation function)
    init(numInputs : Int, layerDefinition: (numNodes: Int, activation: NeuralActivationFunction))
    {
        nodes = []
        for _ in 0..<layerDefinition.numNodes {
            nodes.append(NeuralNode(numInputs: numInputs, activationFunction: layerDefinition.activation))
        }
    }
    
    func getLayerOutputs(inputs: [Double]) -> [Double]
    {
        var outputs : [Double] = []
        //  Assume input array already has bias constant 1.0 appended
        //  Fully-connected nodes means all nodes get the same input array
        if (nodes[0].activation == .SoftMax) {
            var sum = 0.0
            for node in nodes {     //  Sum each output
                sum += node.getNodeOutput(inputs)
            }
            let scale = 1.0 / sum       //  Do division once for efficiency
            for node in nodes {     //  Get the outputs scaled by the sum to give the probability distribuition for the output
                node.lastOutput *= scale
                outputs.append(node.lastOutput)
            }
        }
        else {
            for node in nodes {
                outputs.append(node.getNodeOutput(inputs))
            }
        }
        
        return outputs
    }
    
    func getFinalLayerDelta(expectedOutputs: [Double])
    {
        for nodeIndex in 0..<nodes.count {
            nodes[nodeIndex].getFinalNodeDelta(expectedOutputs[nodeIndex])
        }
    }
    
    func getLayerDelta(nextLayer: NeuralLayer)
    {
        for nNodeIndex in 0..<nodes.count {
            nodes[nNodeIndex].resetDelta()
            //  Add each portion from the nodes in the next forward layer
            for forwardNode in nextLayer.nodes {
                nodes[nNodeIndex].addToDelta(forwardNode.getWeightTimesDelta(nNodeIndex))
            }
            nodes[nNodeIndex].multiplyDeltaByNonLinearityDerivitive()
        }
    }
    
    func updateWeights(inputs: [Double], trainingRate: Double, weightDecay: Double) -> [Double]
    {
        var outputs : [Double] = []
        //  Assume input array already has bias constant 1.0 appended
        //  Fully-connected nodes means all nodes get the same input array
        for node in nodes {
            if (weightDecay < 1) { node.decayWeights(weightDecay) }
            outputs.append(node.updateWeights(inputs, trainingRate: trainingRate))
        }
        
        return outputs
    }
    
    func clearWeightChanges()
    {
        for node in nodes {
            node.clearWeightChanges()
        }
    }
    
    func appendWeightChanges(inputs: [Double]) -> [Double]
    {
        var outputs : [Double] = []
        //  Assume input array already has bias constant 1.0 appended
        //  Fully-connected nodes means all nodes get the same input array
        for node in nodes {
            outputs.append(node.appendWeightChanges(inputs))
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
}

public class NeuralNetwork: Classifier, Regressor {
    //  Layers
    let numInputs : Int
    var layers : [NeuralLayer]
    var trainingRate = 0.3
    var weightDecay = 1.0
    
    ///  Create the neural network based on an array of tuples, one for each non-input layer (number of nodes, activation function)
    ///     There must be at least two layers (hidden layer and output layer), but can have more
    ///     The input layer is defined by the number of inputs only.  The network is fully connected, including a bias term
    ///     If being used for classification, have the output (last) layer have the number of nodes equal to the number of classes
    public init(numInputs : Int, layerDefinitions: [(numNodes: Int, activation: NeuralActivationFunction)])
    {
        self.numInputs = numInputs
        layers = []
        var numInputsFromPreviousLayer = numInputs
        for layerDefinition in layerDefinitions {
            let layer = NeuralLayer(numInputs: numInputsFromPreviousLayer, layerDefinition: layerDefinition)
            layers.append(layer)
            numInputsFromPreviousLayer = layerDefinition.numNodes
        }
    }
    
    public func feedForward(inputs: [Double]) -> [Double] {
        var layerInputs = inputs
        
        //  Go through each layer
        for layer in layers {
            //  Add a bias constant 1.0 to the input array
            layerInputs.append(1.0)
            
            //  Calculate the outputs from the layer
            layerInputs = layer.getLayerOutputs(layerInputs)
        }
        
        return layerInputs
    }
    
    public func trainClassifier(trainData: DataSet) throws
    {
        let epochCount = trainData.size * 2
        let epochSize = trainData.size / 10
        try classificationSGDBatchTrain(trainData, epochSize: epochSize, epochCount : epochCount, trainingRate: trainingRate, weightDecay: weightDecay)
    }
    
    ///  Return a 0 based index of the best neuron output (giving the most probable class for the input)
    public func classifyOne(inputs: [Double]) -> Int {
        //  Get the network outputs
        let outputs = feedForward(inputs)
        
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
        for index in 0..<testData.size {
            testData.classes!.append(classifyOne(testData.inputs[index]))
        }
    }
    
    public func trainRegressor(trainData: DataSet) throws
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
        if (testData.outputDimension != layers.last?.nodes.count) { throw DataTypeError.WrongDimensionOnOutput }
        
        //  predict on each input
        testData.outputs = []
        for index in 0..<testData.size {
            testData.outputs!.append(predictOne(testData.inputs[index]))
        }
    }

    
    public func trainOne(inputs: [Double], expectedOutputs: [Double], trainingRate: Double, weightDecay: Double)
    {
        //  Get the results of a feedForward run (each node remembers its own output)
        feedForward(inputs)
        
        //  Calculate the delta for the final layer
        layers.last!.getFinalLayerDelta(expectedOutputs)
        
        //  Get the deltas for the other layers
        if (layers.count > 1) {
            for nLayerIndex in (layers.count - 2).stride(through: 0, by: -1)
            {
                layers[nLayerIndex].getLayerDelta(layers[nLayerIndex+1])
            }
        }
        
        //  Set the inputs for calculating the weight changes
        var layerInputs = inputs
        
        //  Go through each layer
        for layer in layers {
            //  Add a bias constant 1.0 to the input array
            layerInputs.append(1.0)
            
            //  Calculate the outputs from the layer
            layerInputs = layer.updateWeights(layerInputs, trainingRate: trainingRate, weightDecay: weightDecay)
        }
    }
    
    public func batchTrain(trainData: DataSet, epochIndices : [Int], trainingRate: Double, weightDecay: Double)
    {
        //  Clear the weight change accumulations
        for layer in layers {
            layer.clearWeightChanges()
        }
        
        //  Iterate through each training datum in the batch
        for dataIndex in 0..<epochIndices.count {
            //  Get the results of a feedForward run (each node remembers its own output)
            feedForward(trainData.inputs[epochIndices[dataIndex]])
            
            //  Calculate the delta for the final layer
            layers.last!.getFinalLayerDelta(trainData.outputs![epochIndices[dataIndex]])
            
            //  Get the deltas for the other layers
            if (layers.count > 1) {
                for nLayerIndex in (layers.count - 2).stride(through: 0, by: -1)
                {
                    layers[nLayerIndex].getLayerDelta(layers[nLayerIndex+1])
                }
            }
            
            //  Set the inputs for calculating the weight changes
            var layerInputs = trainData.inputs[epochIndices[dataIndex]]
            
            //  Go through each layer
            for layer in layers {
                //  Add a bias constant 1.0 to the input array
                layerInputs.append(1.0)
                
                //  Append the weight changes for the layer
                layerInputs = layer.appendWeightChanges(layerInputs)
            }
        }
        
        //  Update the weights based on the weight change accumulations
        let averageTrainingRate = trainingRate * -1.0 / Double(epochIndices.count)  //  Make negative so we can use DSP to vectorize equation
        for layer in layers {
            layer.updateWeightsFromAccumulations(averageTrainingRate, weightDecay: weightDecay)
        }
    }

    ///  Train a network on a set of data
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
    
    ///  Train a classification network for a single instance
    public func classificationTrainOne(inputs: [Double], expectedOutput: Int, trainingRate: Double, weightDecay: Double)
    {
        //  Get the false level for the output layer
        var falseLevel = 0.0
        if (layers.last!.nodes[0].activation == .HyberbolicTangent) {falseLevel = -1.0}
        
        //  Create the expected output array for expected class
        var expectedOutputs : [Double] = []
        for classIndex in 0..<layers.last!.nodes.count {
            expectedOutputs.append(classIndex == expectedOutput ? 1.0 : falseLevel)
        }
        
        //  Do the normal training
        trainOne(inputs, expectedOutputs: expectedOutputs, trainingRate: trainingRate, weightDecay: weightDecay)
    }
    
    ///  Train a classification network on a set of data
    public func classificationSGDBatchTrain(trainData: DataSet, epochSize: Int, epochCount : Int, trainingRate: Double, weightDecay: Double) throws
    {
        //  Verify the data set is the right type
        if (trainData.dataType != .Classification) { throw DataTypeError.InvalidDataType }
        
        //  Get the false level for the output layer
        var falseLevel = 0.0
        if (layers.last!.nodes[0].activation == .HyberbolicTangent) {falseLevel = -1.0}
        
        //  Create the expected output array for expected class
        trainData.outputs = [[]]
        var sampleIndex = 0
        for classValue in trainData.classes! {
            trainData.outputs!.append([])
            for classIndex in 0..<layers.last!.nodes.count {
                trainData.outputs![sampleIndex].append(classIndex == classValue ? 1.0 : falseLevel)
            }
            sampleIndex++
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
