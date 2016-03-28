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

    func testSingleNode() {
        //  Create a 1 node network
        var network = NeuralNetwork(numInputs: 1, layerDefinitions: [(numNodes: 1, activation: NeuralActivationFunction.Sigmoid)])
        
        //  Train to output a constant 0.5
        for _ in 0..<1000 {
            network.trainOne([Double(arc4random()) / Double(UInt32.max)], expectedOutputs: [0.5], trainingRate: 1, weightDecay: 0.98)
        }
        
        //  Verify the result
        var result = network.feedForward([Double(arc4random()) / Double(UInt32.max)])
        XCTAssert(fabs(result[0] - 0.5) < 0.02, "network trained to constant")
        
        //  Create a 1 node network
        network = NeuralNetwork(numInputs: 1, layerDefinitions: [(numNodes: 1, activation: NeuralActivationFunction.RectifiedLinear)])
        
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
        let network = NeuralNetwork(numInputs: 2, layerDefinitions: [(numNodes: 1, activation: NeuralActivationFunction.HyberbolicTangent)])
        
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
        let network = NeuralNetwork(numInputs: 2, layerDefinitions: [(numNodes: 1, activation: NeuralActivationFunction.Sigmoid)])
        
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
    
    func testNetworkXORNode() {
        //  Create a 1 hidden layer - 2 node network, with one output node, using two inputs
        let layerDefs = [(numNodes: 2, activation: NeuralActivationFunction.HyberbolicTangent),
                         (numNodes: 1, activation: NeuralActivationFunction.HyberbolicTangent)]
        let network = NeuralNetwork(numInputs: 2, layerDefinitions: layerDefs)
        
        //  Train the network for exclusive or
        for _ in 0..<1000 {
            let input1 = (arc4random() > (UInt32.max >> 1)) ? 1.0 : -1.0
            let input2 = (arc4random() > (UInt32.max >> 1)) ? 1.0 : -1.0
            let result = ((input1 < 0.0 && input2 >= 0.0) ||
                          (input1 >= 0.0 && input2 < 0.0)) ? 1.0 : -1.0
             network.trainOne([input1, input2], expectedOutputs: [result], trainingRate: 0.3, weightDecay: 1.0)
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
    
    func testNetworkBatchTraining() {
        //  Create a 1 hidden layer - 2 node network, with one output node, using two inputs
        let layerDefs = [(numNodes: 2, activation: NeuralActivationFunction.HyberbolicTangent),
            (numNodes: 1, activation: NeuralActivationFunction.HyberbolicTangent)]
        let network = NeuralNetwork(numInputs: 2, layerDefinitions: layerDefs)
        
        //  Create a data set of 4 training examples for the XOR function
        let data = DataSet(dataType: .Regression, inputDimension: 2, outputDimension: 1)
        do {
            try data.addDataPoint(input: [-1.0, -1.0], output:[-1.0])
            try data.addDataPoint(input: [1.0, -1.0], output:[1.0])
            try data.addDataPoint(input: [-1.0, 1.0], output:[1.0])
            try data.addDataPoint(input: [1.0, 1.0], output:[-1.0])
        }
        catch {
            print("Invalid data set created")
        }
        
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
        let layerDefs = [(numNodes: 3, activation: NeuralActivationFunction.Sigmoid),
            (numNodes: 3, activation: NeuralActivationFunction.SigmoidWithCrossEntropy)]
        let network = NeuralNetwork(numInputs: 2, layerDefinitions: layerDefs)
        
        //  Train the network for
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

    func testPerformanceExample() {
        // This is an example of a performance test case.
        self.measureBlock {
            // Put the code you want to measure the time of here.
        }
    }

}
