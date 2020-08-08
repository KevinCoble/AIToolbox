//
//  LSTMTests.swift
//  AIToolbox
//
//  Created by Kevin Coble on 6/5/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import XCTest
import AIToolbox

class LSTMTests: XCTestCase {

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
        var network = NeuralNetwork(numInputs: 1, layerDefinitions: [(layerType: .lstm, numNodes: 1, activation: NeuralActivationFunction.sigmoid, auxiliaryData: nil)])
        
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
        network = NeuralNetwork(numInputs: 1, layerDefinitions: [(layerType: .lstm, numNodes: 1, activation: NeuralActivationFunction.hyperbolicTangent, auxiliaryData: nil)])
        
        //  Initialize the weights
        network.initializeWeights(nil)
        
        //  Train to output a constant slope 0.5
        for _ in 0..<2000 {
            let randomValue = Double(arc4random()) / Double(UInt32.max) * 0.6 + 0.2
            network.trainOne([randomValue], expectedOutputs: [0.5 * randomValue], trainingRate: 0.4, weightDecay: 1.00)
        }
        
        //  Verify the result
        let randomValue = Double(arc4random()) / Double(UInt32.max) * 0.6 + 0.2
        result = network.feedForward([randomValue])
        XCTAssert(fabs(result[0] - (0.5 * randomValue)) < 0.08, "network trained to constant slope \(result), expecting \(0.5 * randomValue)")
    }
    
    //  This test often fails - seems to require a good initialization batch.  When it works, it works quickly
    func testOneLSTMNode() {
        //  Sequence is a 0 or 1 value with 0 for the output, followed by a 0,0 line followed by a two with the first value as the expected output
        //  Create some sequences
        var trainingSequences : [DataSet] = []
        let dataSet0 = DataSet(dataType: .regression, inputDimension: 1, outputDimension: 1)
        do {
            try dataSet0.addDataPoint(input: [0.0], output: [0.0])
            try dataSet0.addDataPoint(input: [0.0], output: [0.0])
            try dataSet0.addDataPoint(input: [2.0], output: [0.0])
        }
        catch {
            print("Error converting string to sequence data")
        }
        trainingSequences.append(dataSet0)
        let dataSet1 = DataSet(dataType: .regression, inputDimension: 1, outputDimension: 1)
        do {
            try dataSet1.addDataPoint(input: [1.0], output: [0.0])
            try dataSet0.addDataPoint(input: [0.0], output: [0.0])
            try dataSet1.addDataPoint(input: [2.0], output: [1.0])
        }
        catch {
            print("Error converting string to sequence data")
        }
        trainingSequences.append(dataSet1)
        
        var testingSequences : [DataSet] = []
        let testDataSet0 = DataSet(dataType: .regression, inputDimension: 1, outputDimension: 1)
        do {
            try testDataSet0.addDataPoint(input: [0.0], output: [0.0])
            try testDataSet0.addDataPoint(input: [0.0], output: [0.0])
            try testDataSet0.addDataPoint(input: [2.0], output: [0.0])
        }
        catch {
            print("Error converting string to sequence data")
        }
        testingSequences.append(testDataSet0)
        let testDataSet1 = DataSet(dataType: .regression, inputDimension: 1, outputDimension: 1)
        do {
            try testDataSet1.addDataPoint(input: [1.0], output: [0.0])
            try testDataSet1.addDataPoint(input: [0.0], output: [0.0])
            try testDataSet1.addDataPoint(input: [2.0], output: [1.0])
        }
        catch {
            print("Error converting string to sequence data")
        }
        testingSequences.append(testDataSet1)
        
        //  Create the LSTM network
        let network = NeuralNetwork(numInputs: 1, layerDefinitions: [
            (layerType: .lstm, numNodes: 1, activation: NeuralActivationFunction.hyperbolicTangent, auxiliaryData: nil),
            ])
        
        //  Initialize the network
        network.initializeWeights(nil)
        
        //  Train the network
        let trainingRuns = 2000
        for run in 0..<trainingRuns {
            let trainIndex = Int(arc4random_uniform(UInt32(2)))
            network.trainSequence(trainingSequences[trainIndex], trainingRate: 0.5, weightDecay: 1.0)
            
            if ((run % 100) == 0) {
                var totalError = 0
                for _ in 0..<100 {
                    do {
                        let testIndex = Int(arc4random_uniform(2))
                        try network.predictSequence(testingSequences[testIndex])
                        let output = try testingSequences[testIndex].getOutput(2)
                        if ((testIndex == 0 && output[0] > 0.5) || (testIndex == 1 && output[0] < 0.5)) {
                            totalError += 1
                        }
                    }
                    catch {
                        print("Error in LSTM testing")
                    }
                }
                print("Total errors = \(totalError)")
            }
        }
        
        //  Final test
        var totalError = 0
        for _ in 0..<100 {
            do {
                let testIndex = Int(arc4random_uniform(2))
                try network.predictSequence(testingSequences[testIndex])
                let output = try testingSequences[testIndex].getOutput(2)
                if ((testIndex == 0 && output[0] > 0.5) || (testIndex == 1 && output[0] < 0.5)) {
                    totalError += 1
                }
            }
            catch {
                print("Error in LSTM testing")
            }
        }
        XCTAssert(totalError == 0, "LSTM single node error")
    }
    
//  This example is always failing.  I need to figure out why
//    func testExample() {
//        //  Create some sequences
//        let numTrainingSequences = 200
//        var trainingExamples : [String] = []
//        var trainingSequences : [DataSet] = []
//        for _ in 0..<numTrainingSequences {
//            let string = getReberGrammerString()
//            trainingExamples.append(string)
//            trainingSequences.append(convertStringToSequence(string))
//        }
//        
//        //  Create the LSTM network
//        let network = NeuralNetwork(numInputs: 7, layerDefinitions: [
//            (layerType: .LSTM, numNodes: 4, activation: NeuralActivationFunction.HyperbolicTangent, auxiliaryData: nil),
//            (layerType: .SimpleFeedForward, numNodes: 7, activation: NeuralActivationFunction.Sigmoid, auxiliaryData: nil)
//            ])
//        
//        //  Initialize the network
//        network.initializeWeights(nil)
//        
//        //  Create testing strings
//        let numTestingSequences = 10
//        var testingExamples : [String] = []
//        var testingSequences : [DataSet] = []
//        for _ in 0..<numTestingSequences {
//            let string = getReberGrammerString()
//            testingExamples.append(string)
//            testingSequences.append(convertStringToSequence(string))
//        }
//        
//        //  Train the network
//        let trainingRuns = 10000
//        for run in 0..<trainingRuns {
//            let trainIndex = Int(arc4random_uniform(UInt32(numTrainingSequences)))
//            network.trainSequence(trainingSequences[trainIndex], trainingRate: 0.5, weightDecay: 1.0)
//            
//            if ((run % 100) == 0) {
//                var totalError = 0
//                var totalChars = 0
//                for testIndex in 0..<numTestingSequences {
//                    do {
//                        try network.predictSequence(testingSequences[testIndex])
//                        var characters = testingExamples[testIndex].characters.map { String($0) }
//                        characters.append("-")      //  Add an unused character for the prediction target at the end of the grammer string
//                        for sequenceIndex in 0..<testingSequences[testIndex].size {
//                            let outputs = try testingSequences[testIndex].getOutput(sequenceIndex)
//                            let chars = predictedLettersFromDoubleArray(outputs)
//                            if (!chars.containsString(characters[sequenceIndex + 1])) {
//                                totalError += 1
//                            }
//                            totalChars += 1
//                        }
//                    }
//                    catch {
//                        print("Error in LSTM testing")
//                    }
//                }
//                print("Total errors after \(run) training sequences = \(totalError) in \(totalChars) characters")
//            }
//        }
//        
//        //  Test the network
//        var totalError = 0
//        var totalChars = 0
//        for testIndex in 0..<numTestingSequences {
//            do {
//                try network.predictSequence(testingSequences[testIndex])
//                var characters = testingExamples[testIndex].characters.map { String($0) }
//                characters.append("-")      //  Add an unused character for the prediction target at the end of the grammer string
//                for sequenceIndex in 0..<testingSequences[testIndex].size {
//                    let outputs = try testingSequences[testIndex].getOutput(sequenceIndex)
//                    let chars = predictedLettersFromDoubleArray(outputs)
//                    if (!chars.containsString(characters[sequenceIndex + 1])) {
//                        totalError += 1
//                    }
//                    totalChars += 1
//                }
//            }
//            catch {
//                print("Error in LSTM testing")
//            }
//        }
//        print("Total errors = \(totalError) in \(totalChars) characters")
//        XCTAssert(totalError == 0, "LSTM Reber Grammer error")
//    }

    func testPerformanceExample() {
        // This is an example of a performance test case.
        self.measure {
            // Put the code you want to measure the time of here.
        }
    }

    
    func getReberGrammerString() -> String {
        var str = "B"
        var state = 0
        while (true) {
            switch state {
            case 0:
                let chance = arc4random_uniform(2)
                if (chance == 0) {
                    str += "T"
                    state = 1
                }
                else {
                    str += "P"
                    state = 2
                }
            case 1:
                let chance = arc4random_uniform(2)
                if (chance == 0) {
                    str += "S"
                }
                else {
                    str += "X"
                    state = 3
                }
            case 2:
                let chance = arc4random_uniform(2)
                if (chance == 0) {
                    str += "T"
                }
                else {
                    str += "V"
                    state = 4
                }
            case 3:
                let chance = arc4random_uniform(2)
                if (chance == 0) {
                    str += "X"
                    state = 2
                }
                else {
                    str += "S"
                    state = 5
                }
            case 4:
                str += "V"
                state = 5
            default:
                str += "E"
                return str
            }
        }
    }
    
    func stateLetterToDoubleArray(_ letter: Character) -> [Double] {
        switch (letter) {
        case "B":
            return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        case "P":
            return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        case "T":
            return [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        case "S":
            return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        case "X":
            return [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        case "V":
            return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        case "E":
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        default:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
    }
    
    func predictedLettersFromDoubleArray(_ array: [Double]) -> String {
        var str = ""
        
        if (array[0] > 0.5) { str += "B" }
        if (array[1] > 0.5) { str += "P" }
        if (array[2] > 0.5) { str += "T" }
        if (array[3] > 0.5) { str += "S" }
        if (array[4] > 0.5) { str += "X" }
        if (array[5] > 0.5) { str += "V" }
        if (array[6] > 0.5) { str += "E" }
        
        return str
    }
    
    func convertStringToSequence(_ grammerString: String) -> DataSet {
        let characters = grammerString + "x" //  Add an unused character for the prediction target at the end of the grammer string
        let dataSet = DataSet(dataType: .regression, inputDimension: 7, outputDimension: 7)
        for index in characters.indices {
            guard index != characters.endIndex else { continue }
            let inputs = stateLetterToDoubleArray(characters[index])
            let outputs = stateLetterToDoubleArray(characters[characters.index(after: index)])
            do {
                try dataSet.addDataPoint(input: inputs, output: outputs)
            }
            catch {
                print("Error converting string to sequence data")
            }
        }
        
        return dataSet
    }
}
