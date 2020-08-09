//
//  MetalNeuralNetworkTests.swift
//  AIToolbox
//
//  Created by Kevin Coble on 1/10/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import XCTest
import AIToolbox

@available(OSX 10.11, iOS 8.0, *)
class MetalNeuralNetworkTests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }

    func testSingleNode() {
        //  Test a single node
        if let network = MetalNeuralNetwork(numInputs: 1, layerDefinitions: [(numNodes: 1, activation: NeuralActivationFunction.sigmoid)]) {
            
            //  Train to output a constant 0.5
            for _ in 0..<1000 {
                network.trainOne([Float(arc4random()) / Float(UInt32.max)], expectedOutputs: [0.5], trainingRate: 1, weightDecay: 0.98)
            }
            
            //  Verify the result
            let result = network.feedForward([Float(arc4random()) / Float(UInt32.max)])
            XCTAssert(abs(result[0] - 0.5) < 0.02, "network trained to constant")
        }
        
            //  Create a 1 node network
        if let network = MetalNeuralNetwork(numInputs: 1, layerDefinitions: [(numNodes: 1, activation: NeuralActivationFunction.rectifiedLinear)]) {
            
            //  Train to output a constant slope 0.5
            for _ in 0..<1000 {
                let randomValue = Float(arc4random()) / Float(UInt32.max) * 0.6 + 0.2
                network.trainOne([randomValue], expectedOutputs: [0.5 * randomValue], trainingRate: 0.4, weightDecay: 0.99)
            }
            
            //  Verify the result
            let randomValue = Float(arc4random()) / Float(UInt32.max) * 0.6 + 0.2
            let result = network.feedForward([randomValue])
            XCTAssert(abs(result[0] - (0.5 * randomValue)) < 0.08, "network trained to constant slope \(result), expecting \(0.5 * randomValue)")
        }
    }
}
