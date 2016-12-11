//
//  NeuralNetworkTests.swift
//  AIToolbox
//
//  Created by Kevin Coble on 10/19/15.
//  Copyright © 2015 Kevin Coble. All rights reserved.
//

import XCTest
import AIToolbox

class NeuralNetworkTests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testGradients() {
        var network = NeuralNetwork(numInputs: 1, layerDefinitions: [(layerType: .simpleFeedForward, numNodes: 1, activation: NeuralActivationFunction.sigmoid, auxiliaryData: nil)])
        network.initializeWeights(nil)
        var result = network.gradientCheck(inputs: [1.0], expectedOutputs: [0.0], ε: 0.0001, Δ: 0.00001)
        XCTAssert(result, "Gradient check sigmoid single node")

        network = NeuralNetwork(numInputs: 1, layerDefinitions: [(layerType: .simpleFeedForward, numNodes: 1, activation: NeuralActivationFunction.hyperbolicTangent, auxiliaryData: nil)])
        network.initializeWeights(nil)
        result = network.gradientCheck(inputs: [1.0], expectedOutputs: [0.0], ε: 0.0001, Δ: 0.00001)
        XCTAssert(result, "Gradient check tanh single node")
        
        network = NeuralNetwork(numInputs: 1, layerDefinitions: [(layerType: .simpleFeedForward, numNodes: 1, activation: NeuralActivationFunction.rectifiedLinear, auxiliaryData: nil)])
        network.initializeWeights(nil)
        result = network.gradientCheck(inputs: [1.0], expectedOutputs: [0.0], ε: 0.0001, Δ: 0.00001)
        XCTAssert(result, "Gradient check ReLu single node")
        
        network = NeuralNetwork(numInputs: 1, layerDefinitions: [(layerType: .simpleFeedForward, numNodes: 1, activation: NeuralActivationFunction.none, auxiliaryData: nil)])
        network.initializeWeights(nil)
        result = network.gradientCheck(inputs: [1.0], expectedOutputs: [0.0], ε: 0.0001, Δ: 0.00001)
        XCTAssert(result, "Gradient check none single node")
        
        //  Create a 2 hidden layer - 2 node network, with one output node, using two inputs
        var layerDefs : [(layerType: NeuronLayerType, numNodes: Int, activation: NeuralActivationFunction, auxiliaryData: AnyObject?)] =
            [(layerType: .simpleFeedForward, numNodes: 2, activation: NeuralActivationFunction.hyperbolicTangent, auxiliaryData: nil),
             (layerType: .simpleFeedForward, numNodes: 2, activation: NeuralActivationFunction.hyperbolicTangent, auxiliaryData: nil),
             (layerType: .simpleFeedForward, numNodes: 1, activation: NeuralActivationFunction.hyperbolicTangent, auxiliaryData: nil)]
        network = NeuralNetwork(numInputs: 2, layerDefinitions: layerDefs)
        network.initializeWeights(nil)
        result = network.gradientCheck(inputs: [1.0, -1.0], expectedOutputs: [0.0], ε: 0.0001, Δ: 0.00001)
        XCTAssert(result, "Gradient check tanh three layers")
        
        //  Create a 2 hidden layer - 2 node network, with one output node, using two inputs - using layers with node classes
        layerDefs =
            [(layerType: .simpleFeedForwardWithNodes, numNodes: 2, activation: NeuralActivationFunction.hyperbolicTangent, auxiliaryData: nil),
             (layerType: .simpleFeedForwardWithNodes, numNodes: 2, activation: NeuralActivationFunction.hyperbolicTangent, auxiliaryData: nil),
             (layerType: .simpleFeedForwardWithNodes, numNodes: 1, activation: NeuralActivationFunction.hyperbolicTangent, auxiliaryData: nil)]
        network = NeuralNetwork(numInputs: 2, layerDefinitions: layerDefs)
        network.initializeWeights(nil)
        result = network.gradientCheck(inputs: [1.0, -1.0], expectedOutputs: [0.0], ε: 0.0001, Δ: 0.00001)
        XCTAssert(result, "Gradient check tanh three layers with nodes")
        
        //  Create a 1 layer RNN
        network = NeuralNetwork(numInputs: 1, layerDefinitions: [(layerType: .simpleRecurrent, numNodes: 1, activation: NeuralActivationFunction.hyperbolicTangent, auxiliaryData: nil)])
        network.initializeWeights(nil)
        result = network.gradientCheck(inputs: [1.0], expectedOutputs: [0.0], ε: 0.0001, Δ: 0.00001)
        XCTAssert(result, "Gradient check RNN")
        
        //  Create a 1 layer RNN
        network = NeuralNetwork(numInputs: 1, layerDefinitions: [(layerType: .simpleRecurrentWithNodes, numNodes: 1, activation: NeuralActivationFunction.hyperbolicTangent, auxiliaryData: nil)])
        network.initializeWeights(nil)
        result = network.gradientCheck(inputs: [1.0], expectedOutputs: [0.0], ε: 0.0001, Δ: 0.00001)
        XCTAssert(result, "Gradient check RNN with nodes")
    }

    func testSingleNode() {
        //  Create a 1 node network
        var network = NeuralNetwork(numInputs: 1, layerDefinitions: [(layerType: .simpleFeedForward, numNodes: 1, activation: NeuralActivationFunction.sigmoid, auxiliaryData: nil)])
        
        //  Initialize the weights
        network.initializeWeights(nil)
        
        //  Train to output a constant 0.5
        for _ in 0..<1000 {
            network.trainOne([Double(arc4random()) / Double(UInt32.max)], expectedOutputs: [0.5], trainingRate: 1, weightDecay: 0.98)
        }
        network.trainOne([Double(arc4random()) / Double(UInt32.max)], expectedOutputs: [0.5], trainingRate: 1, weightDecay: 1.00)
        
        //  Verify the result
        var result = network.feedForward([Double(arc4random()) / Double(UInt32.max)])
        XCTAssert(fabs(result[0] - 0.5) < 0.02, "network trained to constant")
        
        //  Create a 1 node network
        network = NeuralNetwork(numInputs: 1, layerDefinitions: [(layerType: .simpleFeedForward, numNodes: 1, activation: NeuralActivationFunction.rectifiedLinear, auxiliaryData: nil)])
        
        //  Initialize the weights
        network.initializeWeights(nil)
        
        //  Train to output a constant slope 0.5
        for _ in 0..<1000 {
            let randomValue = Double(arc4random()) / Double(UInt32.max) * 0.6 + 0.2
            network.trainOne([randomValue], expectedOutputs: [0.5 * randomValue], trainingRate: 0.4, weightDecay: 0.99)
        }
        
        //  Verify the result
        let randomValue = Double(arc4random()) / Double(UInt32.max) * 0.6 + 0.2
        result = network.feedForward([randomValue])
        XCTAssert(fabs(result[0] - (0.5 * randomValue)) < 0.08, "network trained to constant slope \(result), expecting \(0.5 * randomValue)")
    }
    
    func testNetworkBooleanAnd() {
        //  Create a 1 node network
        let network = NeuralNetwork(numInputs: 2, layerDefinitions: [(layerType: .simpleFeedForward, numNodes: 1, activation: NeuralActivationFunction.hyperbolicTangent, auxiliaryData: nil)])
        
        //  Initialize the weights
        network.initializeWeights(nil)
        
        //  Train the network for and function
        for _ in 0..<1000 {
            let input1 = (arc4random() > (UInt32.max >> 1)) ? 1.0 : -1.0
            let input2 = (arc4random() > (UInt32.max >> 1)) ? 1.0 : -1.0
            let result = (input1 >= 0.5 && input2 >= 0.5) ? 1.0 : -1.0
            network.trainOne([input1, input2], expectedOutputs: [result], trainingRate: 0.3, weightDecay: 1.0)
        }
        
        //  Test the network with each posibility
        var result = network.feedForward([-1.0, -1.0])
        XCTAssert(result[0] < -0.5, "network trained for boolean AND -  input [false. false]")
        result = network.feedForward([ 1.0, -1.0])
        XCTAssert(result[0] < -0.5, "network trained for boolean AND -  input [true.  false]")
        result = network.feedForward([-1.0,  1.0])
        XCTAssert(result[0] < -0.5, "network trained for boolean AND -  input [false. true ]")
        result = network.feedForward([ 1.0,  1.0])
        XCTAssert(result[0] >= 0.5, "network trained for boolean AND -  input [true.  true ]")
    }
    
    func testNetworkAnalogAnd() {
        //  Create a 1 node network
        let network = NeuralNetwork(numInputs: 2, layerDefinitions: [(layerType: .simpleFeedForward, numNodes: 1, activation: NeuralActivationFunction.sigmoid, auxiliaryData: nil)])
        
        //  Initialize the weights
        network.initializeWeights(nil)
        
        //  Train the network for and function
        for _ in 0..<1000 {
            var input1 = 0.5
            while (input1 > 0.4 && input1 < 0.6) {
                input1 = Double(arc4random()) / Double(UInt32.max)
            }
            var input2 = 0.5
            while (input2 > 0.4 && input2 < 0.6) {
                input2 = Double(arc4random()) / Double(UInt32.max)
            }
            let result = (input1 >= 0.5 && input2 >= 0.5) ? 1.0 : 0.0
            network.trainOne([input1, input2], expectedOutputs: [result], trainingRate: 0.05, weightDecay: 1.0)
        }
        
        //  Test the network a hundred times
        var correctCount = 0
        for _ in 0..<100 {
            var input1 = 0.5
            while (input1 > 0.4 && input1 < 0.6) {
                input1 = Double(arc4random()) / Double(UInt32.max)
            }
            var input2 = 0.5
            while (input2 > 0.4 && input2 < 0.6) {
                input2 = Double(arc4random()) / Double(UInt32.max)
            }
            let expectedResult = (input1 >= 0.5 && input2 >= 0.5) ? true : false
            let result = network.feedForward([input1, input2])
            let booleanResult = (result[0] >= 0.5) ? true : false
            if booleanResult == expectedResult {correctCount += 1}
        }
        XCTAssert(correctCount >= 70, "network trained for AND")    //  Best case is around 75% accuracy without the deadband.  70 gives us room for bad random data set
    }
    
    func testNetworkXORNode() {     //  This test can sometimes fail because boolean inputs without batch training leads to gradient calculation issues
        //  Create a 1 hidden layer - 2 node network, with one output node, using two inputs
        let layerDefs : [(layerType: NeuronLayerType, numNodes: Int, activation: NeuralActivationFunction, auxiliaryData: AnyObject?)] =
            [(layerType: .simpleFeedForward, numNodes: 2, activation: NeuralActivationFunction.hyperbolicTangent, auxiliaryData: nil),
             (layerType: .simpleFeedForward, numNodes: 2, activation: NeuralActivationFunction.hyperbolicTangent, auxiliaryData: nil),
             (layerType: .simpleFeedForward, numNodes: 1, activation: NeuralActivationFunction.hyperbolicTangent, auxiliaryData: nil)]
        let network = NeuralNetwork(numInputs: 2, layerDefinitions: layerDefs)
        
        //  Initialize the weights
        network.initializeWeights(nil)
        
        //  Train the network for exclusive or
        var trainingRate = 0.5
        for index in 0..<5000 {
            let input1 = (arc4random() > (UInt32.max >> 1)) ? 1.0 : -1.0
            let input2 = (arc4random() > (UInt32.max >> 1)) ? 1.0 : -1.0
            let result = ((input1 < 0.0 && input2 >= 0.0) ||
                          (input1 >= 0.0 && input2 < 0.0)) ? 1.0 : -1.0
             network.trainOne([input1, input2], expectedOutputs: [result], trainingRate: trainingRate, weightDecay: 1.0)
            if ((index % 1000) == 0) { trainingRate *= 0.5 }
        }
        
        //  Test the network with each posibility
        var result = network.feedForward([-1.0, -1.0])
        XCTAssert(result[0] < 0.0, "network trained for boolean and XOR -  input [false, false]")
        result = network.feedForward([ 1.0, -1.0])
        XCTAssert(result[0] >= 0.0, "network trained for boolean and XOR -  input [true,  false]")
        result = network.feedForward([-1.0,  1.0])
        XCTAssert(result[0] >= 0.0, "network trained for boolean and XOR -  input [false, true ]")
        result = network.feedForward([ 1.0,  1.0])
        XCTAssert(result[0] < 0.0, "network trained for boolean and XOR -  input [true,  true ]")
    }
    
    func testRMSProp() {
        //  Create a 1 node network
        let network = NeuralNetwork(numInputs: 2, layerDefinitions: [(layerType: .simpleFeedForwardWithNodes, numNodes: 1, activation: NeuralActivationFunction.hyperbolicTangent, auxiliaryData: nil)])
        network.setNeuralWeightUpdateMethod(.rmsProp, 0.98)
        
        //  Initialize the weights
        network.initializeWeights(nil)
        
        //  Train the network for and function
        for _ in 0..<1000 {
            let input1 = (arc4random() > (UInt32.max >> 1)) ? 1.0 : -1.0
            let input2 = (arc4random() > (UInt32.max >> 1)) ? 1.0 : -1.0
            let result = (input1 >= 0.5 && input2 >= 0.5) ? 1.0 : -1.0
            network.trainOne([input1, input2], expectedOutputs: [result], trainingRate: 0.3, weightDecay: 1.0)
        }
        
        //  Test the network with each posibility
        var result = network.feedForward([-1.0, -1.0])
        XCTAssert(result[0] < -0.5, "network trained for boolean AND -  input [false. false]")
        result = network.feedForward([ 1.0, -1.0])
        XCTAssert(result[0] < -0.5, "network trained for boolean AND -  input [true.  false]")
        result = network.feedForward([-1.0,  1.0])
        XCTAssert(result[0] < -0.5, "network trained for boolean AND -  input [false. true ]")
        result = network.feedForward([ 1.0,  1.0])
        XCTAssert(result[0] >= 0.5, "network trained for boolean AND -  input [true.  true ]")
    }
    
    func testNetworkBatchTraining() {
        //  Create a 2 hidden layer - 2 node network, with one output node, using two inputs
        let layerDefs : [(layerType: NeuronLayerType, numNodes: Int, activation: NeuralActivationFunction, auxiliaryData: AnyObject?)] =
            [(layerType: .simpleFeedForward, numNodes: 2, activation: NeuralActivationFunction.hyperbolicTangent, auxiliaryData: nil),
             (layerType: .simpleFeedForward, numNodes: 2, activation: NeuralActivationFunction.hyperbolicTangent, auxiliaryData: nil),
             (layerType: .simpleFeedForward, numNodes: 1, activation: NeuralActivationFunction.hyperbolicTangent, auxiliaryData: nil)]
        let network = NeuralNetwork(numInputs: 2, layerDefinitions: layerDefs)
        
        //  Create a data set of 4 training examples for the XOR function
        let data = DataSet(dataType: .regression, inputDimension: 2, outputDimension: 1)
        do {
            try data.addDataPoint(input: [-1.0, -1.0], output:[-1.0])
            try data.addDataPoint(input: [1.0, -1.0], output:[1.0])
            try data.addDataPoint(input: [-1.0, 1.0], output:[1.0])
            try data.addDataPoint(input: [1.0, 1.0], output:[-1.0])
        }
        catch {
            print("Invalid data set created")
        }
        
        //  Initialize the weights
        network.initializeWeights(data)
        
        //  Train the network for 1000 epochs of 4
        var trainIndexes = [0, 1, 2, 4]
        for _ in 0..<1000 {
            //  Get a random batch
            for index in 0..<trainIndexes.count {
                trainIndexes[index] = Int(arc4random_uniform(UInt32(data.size)))
            }
            //  Train the batch
            network.batchTrain(data, epochIndices: trainIndexes, trainingRate: 0.5, weightDecay: 1.0)
        }
        
        //  Test the network with each posibility
        var result = network.feedForward([-1.0, -1.0])
        XCTAssert(result[0] < 0.0, "network trained for boolean and XOR -  input [false, false]")
        result = network.feedForward([ 1.0, -1.0])
        XCTAssert(result[0] >= 0.0, "network trained for boolean and XOR -  input [true,  false]")
        result = network.feedForward([-1.0,  1.0])
        XCTAssert(result[0] >= 0.0, "network trained for boolean and XOR-  input [false, true ]")
        result = network.feedForward([ 1.0,  1.0])
        XCTAssert(result[0] < 0.0, "network trained for boolean and XOR -  input [true,  true ]")
    }
    
    func testClassification() {
        //  Three classes with two inputs
        //  Class 0 - centers on input 1 = 0.2, input 2 = 0.3
        //  Class 1 - centers on input 1 = 0.8, input 2 = 0.2
        //  Class 2 - centers on input 1 = 0.6, input 2 = 0.7
        //  Create a 1 hidden layer - 3 node network, with three output nodes, using two inputs
        let layerDefs : [(layerType: NeuronLayerType, numNodes: Int, activation: NeuralActivationFunction, auxiliaryData: AnyObject?)] =
            [(layerType: .simpleFeedForward, numNodes: 3, activation: NeuralActivationFunction.sigmoid, auxiliaryData: nil),
             (layerType: .simpleFeedForward, numNodes: 3, activation: NeuralActivationFunction.sigmoidWithCrossEntropy, auxiliaryData: nil)]
        let network = NeuralNetwork(numInputs: 2, layerDefinitions: layerDefs)
        
        //  Initialize the weights
        network.initializeWeights(nil)
        
        //  Train the network for classification
        var input1 : Double
        var input2 : Double
        
        for _ in 0..<1000 {
            let classNumber = Int(arc4random_uniform(3))
            if (classNumber == 0) {
                input1 = 0.2
                input2 = 0.3
            }
            else if (classNumber == 1) {
                input1 = 0.8
                input2 = 0.2
            }
            else {
                input1 = 0.6
                input2 = 0.7
            }
            input1 += (Double(arc4random()) * 0.2 / Double(UInt32.max)) - 0.1
            input2 += (Double(arc4random()) * 0.2 / Double(UInt32.max)) - 0.1
            network.classificationTrainOne([input1, input2], expectedOutput: classNumber, trainingRate: 0.3, weightDecay: 1.0)
        }
        
        //  Test
        var correctCount = 0
        for _ in 0..<100 {
            let classNumber = Int(arc4random_uniform(3))
            if (classNumber == 0) {
                input1 = 0.2
                input2 = 0.3
            }
            else if (classNumber == 1) {
                input1 = 0.8
                input2 = 0.2
            }
            else {
                input1 = 0.6
                input2 = 0.7
            }
            input1 += (Double(arc4random()) * 0.2 / Double(UInt32.max)) - 0.1
            input2 += (Double(arc4random()) * 0.2 / Double(UInt32.max)) - 0.1
            let result = network.classifyOne([input1, input2])
            if (result == classNumber) {correctCount += 1}
        }
        XCTAssert(correctCount >= 80, "network trained for classification")    //  80 gives us room for bad random data
    }
    
    func testRNNNode() {
        
        //  Create a recurrent net of 1 recurrent node
        let layerDefs : [(layerType: NeuronLayerType, numNodes: Int, activation: NeuralActivationFunction, auxiliaryData: AnyObject?)] =
            [(layerType: .simpleRecurrent, numNodes: 1, activation: NeuralActivationFunction.sigmoid, auxiliaryData: nil)]
        let network = NeuralNetwork(numInputs: 1, layerDefinitions: layerDefs)
        
        //  Initialize the weights of the network
        network.initializeWeights(nil)
        
        //  Create and train on a number of sequences
        var trainingRate = 2.0
        for index in 0..<2000 {
            //  Get three numbers
            let firstNumber = (Double(arc4random()) * 20.0 / Double(UInt32.max)) - 10.0
            let secondNumber = (Double(arc4random()) * 20.0 / Double(UInt32.max)) - 10.0
            let thirdNumber = (Double(arc4random()) * 20.0 / Double(UInt32.max)) - 10.0
            
            //  Create a data set with the sequence.  Output is sigmoid of input plus previous output
            let sequenceData = DataSet(dataType: .regression, inputDimension: 1, outputDimension: 1)
            do {
                let output1 = sigmoid(firstNumber + 0.0)
                try sequenceData.addDataPoint(input: [firstNumber], output: [output1])
                let output2 = sigmoid(secondNumber + output1)
                try sequenceData.addDataPoint(input: [secondNumber], output: [output2])
                try sequenceData.addDataPoint(input: [thirdNumber], output: [sigmoid(thirdNumber + output2)])
            }
            catch {
                print("Invalid sequence data set created")
            }
            
            //  Train on the sequence
            network.trainSequence(sequenceData, trainingRate: trainingRate, weightDecay: 1.0)
            
            //  Lower the training rate every 2000
            if ((index % 2000) == 0) {
                trainingRate *= 0.5
            }
            
            //!!  Test every 100
//            if ((index % 100) == 0) {
//                let sequenceData = DataSet(dataType: .Regression, inputDimension: 1, outputDimension: 1)
//                do {
//                    let firstNumber = (Double(arc4random()) * 20.0 / Double(UInt32.max)) - 10.0
//                    let secondNumber = (Double(arc4random()) * 20.0 / Double(UInt32.max)) - 10.0
//                    let thirdNumber = (Double(arc4random()) * 20.0 / Double(UInt32.max)) - 10.0
//                    try sequenceData.addTestDataPoint(input: [firstNumber])
//                    try sequenceData.addTestDataPoint(input: [secondNumber])
//                    try sequenceData.addTestDataPoint(input: [thirdNumber])
//                    try network.predictSequence(sequenceData)
//                    let output1 = sigmoid(firstNumber + 0.0)
//                    let output2 = sigmoid(secondNumber + output1)
//                    let diff1 = try sequenceData.getOutput(0)[0] - output1
//                    let diff2 = try sequenceData.getOutput(1)[0] - output2
//                    let diff3 = try sequenceData.getOutput(2)[0] - sigmoid(thirdNumber + output2)
//                    print("at index \(index) difference is \(diff1), \(diff2), \(diff3) for values \(firstNumber), \(secondNumber), \(thirdNumber)")
//                }
//                catch {
//                    print("Invalid test sequence data set created")
//                }
//                do {
//                    let weights = try network.getParameters()
//                    print("at index \(index), weights are \(weights)")
//                }
//                catch {
//                    print("error getting weights")
//                }
//            }
        }
        
        //  Test
        let sequenceData = DataSet(dataType: .regression, inputDimension: 1, outputDimension: 1)
        do {
            let firstNumber = (Double(arc4random()) * 20.0 / Double(UInt32.max)) - 10.0
            let secondNumber = (Double(arc4random()) * 20.0 / Double(UInt32.max)) - 10.0
            let thirdNumber = (Double(arc4random()) * 20.0 / Double(UInt32.max)) - 10.0
            try sequenceData.addTestDataPoint(input: [firstNumber])
            try sequenceData.addTestDataPoint(input: [secondNumber])
            try sequenceData.addTestDataPoint(input: [thirdNumber])
            try network.predictSequence(sequenceData)
            let out1 = sigmoid(firstNumber + 0.0)
            let out2 = sigmoid(secondNumber + out1)
            let output0 = try sequenceData.getOutput(0)[0] - out1
            let output1 = try sequenceData.getOutput(1)[0] - out2
            let output2 = try sequenceData.getOutput(2)[0] - sigmoid(thirdNumber + out2)
            XCTAssert(output0 < 0.05, "recurrent node first in sequence")
            XCTAssert((output1 < 0.05), "recurrent node - second in sequence")
            XCTAssert((output2 < 0.05), "recurrent node - third in sequence")
        }
        catch {
            print("Invalid sequence data set created")
        }
    }
    func sigmoid(_ x: Double) -> Double {
        return 1.0 / (1.0 + exp(-x))
    }
    
    func testSimpleRNN() {
        
        //  Create a recurrent net
        let layerDefs : [(layerType: NeuronLayerType, numNodes: Int, activation: NeuralActivationFunction, auxiliaryData: AnyObject?)] =
            [(layerType: .simpleRecurrent, numNodes: 2, activation: NeuralActivationFunction.sigmoid, auxiliaryData: nil),
             (layerType: .simpleFeedForward, numNodes: 1, activation: NeuralActivationFunction.sigmoid, auxiliaryData: nil)]
        let network = NeuralNetwork(numInputs: 1, layerDefinitions: layerDefs)
        
        //  Initialize the weights of the network
        network.initializeWeights(nil)

        //  Create and train on a number of sequences where output is high if the current or last value is > 5
//        for index in 0..<5000 {
        for _ in 0..<10000 {
            //  Create a data set with the sequence
            let sequenceData = DataSet(dataType: .regression, inputDimension: 1, outputDimension: 1)
            do {
                var lastInput = 0.0
                for _ in 0..<10 {
                    let input = (Double(arc4random()) * 20.0 / Double(UInt32.max)) - 10.0
                    let output = (input > 5.0 || lastInput > 5.0) ? 1.0 : 0.0
                    try sequenceData.addDataPoint(input: [input], output: [output])
                    lastInput = input
                }
            }
            catch {
                print("Invalid sequence data set created")
            }
            
            //  Train on the sequence
            network.trainSequence(sequenceData, trainingRate: 1.0, weightDecay: 1.0)
            
//            //  Test every 500
//            if ((index % 500) == 0) {     //
//                let sequenceData = DataSet(dataType: .Regression, inputDimension: 1, outputDimension: 1)
//                do {
//                    try sequenceData.addTestDataPoint(input: [-5.0])        //  Not high
//                    try sequenceData.addTestDataPoint(input: [9.0])         //  High
//                    try sequenceData.addTestDataPoint(input: [0.0])         //  Still High
//                    try sequenceData.addTestDataPoint(input: [0.0])         //  Not high
//                    try network.predictSequence(sequenceData)
//                    let output0 = try sequenceData.getOutput(0)[0]
//                    let output1 = try sequenceData.getOutput(1)[0]
//                    let output2 = try sequenceData.getOutput(2)[0]
//                    let output3 = try sequenceData.getOutput(3)[0]
//                    print("at index \(index) result is \(output0), \(output1), \(output2), \(output3)")
//                }
//                catch {
//                    print("Invalid test sequence data set created")
//                }
//            }
        }
        
        //  Create a test sequence
        
        //  Create a data set with the sequence
        let sequenceData = DataSet(dataType: .regression, inputDimension: 1, outputDimension: 1)
        do {
            try sequenceData.addTestDataPoint(input: [-5.0])        //  Not high
            try sequenceData.addTestDataPoint(input: [9.0])         //  High
            try sequenceData.addTestDataPoint(input: [0.0])         //  Still High
            try sequenceData.addTestDataPoint(input: [0.0])         //  Not high
        }
        catch {
            print("Invalid test sequence data set created")
        }
        
        //  Test the sequence
        do {
            try network.predictSequence(sequenceData)
        }
        catch {
            print("Error predicting sequence")
        }
        
        //  Verify each element matches
        do {
            let output0 = try sequenceData.getOutput(0)[0]
            let output1 = try sequenceData.getOutput(1)[0]
            let output2 = try sequenceData.getOutput(2)[0]
            let output3 = try sequenceData.getOutput(3)[0]
            XCTAssert(output0 < 0.5, "recurrent sequence comparison")
            XCTAssert(output1 > 0.5, "recurrent sequence comparison")
            XCTAssert(output2 > 0.5, "recurrent sequence comparison")
            XCTAssert(output3 < 0.5, "recurrent sequence comparison")
        }
        catch {
            print("Sequence comparison error")
        }
    }

    //  Commented out because it takes a long time.  Example works most of the time (random weight initialization and train vectors can affect performance)
//    func testRNNBinaryAddition() {
//        
//        //  Create a recurrent net
//        let layerDefs : [(layerType: NeuronLayerType, numNodes: Int, activation: NeuralActivationFunction, auxiliaryData: AnyObject?)] =
//            [(layerType: .SimpleRecurrent, numNodes: 16, activation: NeuralActivationFunction.Sigmoid, auxiliaryData: nil),
//             //             (layerType: .SimpleFeedForward, numNodes: 2, activation: NeuralActivationFunction.Sigmoid, auxiliaryData: nil),
//                (layerType: .SimpleFeedForward, numNodes: 1, activation: NeuralActivationFunction.Sigmoid, auxiliaryData: nil)]
//        let network = NeuralNetwork(numInputs: 2, layerDefinitions: layerDefs)
//        
//        //  Initialize the weights of the network
//        network.initializeWeights(nil)
//        
//        //  Create and train on a number of sequences
//        for _ in 0..<10000 {
////        for index in 0..<10000 {        //  index variable needed if periodic reporting is uncommented
//            //  Get two numbers between 0 and 127 and their sum
//            let firstNumber = arc4random_uniform(128)
//            let secondNumber = arc4random_uniform(128)
//            let sum = firstNumber + secondNumber
//            
//            //  Convert the numbers to binary representation in a double array
//            let firstArray = toBinaryArray(firstNumber, length: 8)
//            let secondArray = toBinaryArray(secondNumber, length: 8)
//            let sumArray = toBinaryArray(sum, length: 8)
//            
//            //  Create a data set with the sequence
//            let sequenceData = DataSet(dataType: .Regression, inputDimension: 2, outputDimension: 1)
//            do {
//                for index in 0..<sumArray.count {
//                    try sequenceData.addDataPoint(input: [firstArray[index], secondArray[index]], output: [sumArray[index]])
//                }
//            }
//            catch {
//                print("Invalid sequence data set created")
//            }
//            
//            //  Train on the sequence
//            network.trainSequence(sequenceData, trainingRate: 0.5, weightDecay: 1.0)
//            
//            //!!  Test every 500
////            if ((index % 500) == 0) {
////                let sequenceData = DataSet(dataType: .Regression, inputDimension: 2, outputDimension: 1)
////                do {
////                    try sequenceData.addTestDataPoint(input: [1.0, 1.0])
////                    try sequenceData.addTestDataPoint(input: [0.0, 0.0])
////                    try network.predictSequence(sequenceData)
////                    let output0 = try sequenceData.getOutput(0)[0]
////                    let output1 = try sequenceData.getOutput(1)[0]
////                    print("at index \(index) result is \(output0), \(output1)")
////                }
////                catch {
////                    print("Invalid test sequence data set created")
////                }
////            }
//        }
//        
//        //  Create a test sequence
//        //  Get two numbers between 0 and 127 and their sum
//        let firstNumber = arc4random_uniform(128)
//        let secondNumber = arc4random_uniform(128)
//        let sum = firstNumber + secondNumber
//        
//        //  Convert the numbers to binary representation in a double array
//        let firstArray = toBinaryArray(firstNumber, length: 8)
//        let secondArray = toBinaryArray(secondNumber, length: 8)
//        let sumArray = toBinaryArray(sum, length: 8)
//        
//        //  Create a data set with the sequence
//        let sequenceData = DataSet(dataType: .Regression, inputDimension: 2, outputDimension: 1)
//        do {
//            for index in 0..<firstArray.count {
//                try sequenceData.addTestDataPoint(input: [firstArray[index], secondArray[index]])
//            }
//        }
//        catch {
//            print("Invalid test sequence data set created")
//        }
//        
//        //  Test the sequence
//        do {
//            try network.predictSequence(sequenceData)
//        }
//        catch {
//            print("Error predicting sequence")
//        }
//        
//        //  Verify each element matches
//        do {
//            for index in 0..<sumArray.count {
//                var matches = false
//                let output = try sequenceData.getOutput(index)[0]
//                if (sumArray[index] == 0.0 && output < 0.5) { matches = true }
//                if (sumArray[index] == 1.0 && output > 0.5) { matches = true }
//                XCTAssert(matches, "recurrent sequence comparison")
//            }
//        }
//        catch {
//            print("Sequence comparison error")
//        }
//    }
//    func toBinaryArray(number: UInt32, length: Int) -> [Double]     //  Convert a number to a fix-sized array of 0.0's and 1.0's
//    {
//        let string = String(number, radix: 2)
//        let bitChars = Array(string.characters)
//        var array = [Double](count: length, repeatedValue: 0.0)
//        for (index, item) in bitChars.reverse().enumerate() {
//            if (item != "0") {
//                array[index] = 1.0
//            }
//        }
//        
//        return array
//    }

    func testPerformanceExample() {
        // This is an example of a performance test case.
        self.measure {
            // Put the code you want to measure the time of here.
        }
    }

}
