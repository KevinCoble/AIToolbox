//
//  DeepNetworkTests.swift
//  AIToolbox
//
//  Created by Kevin Coble on 10/16/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import XCTest
import AIToolbox

class DeepNetworkTests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }

    func testSingleNNNode() {
        let deep = DeepNetwork()
        let layer = DeepLayer()
        let channel = DeepChannel(identifier: "X", sourceChannels: ["X"])
        let op = DeepNeuralNetwork(activation: .rectifiedLinear, size: DeepChannelSize(dimensionCount: 1, dimensionValues: [1]))
        let input = DeepNetworkInput(inputID: "X", size: DeepChannelSize(dimensionCount: 1, dimensionValues: [1]), values: [0.0])
        channel.addNetworkOperator(op)
        layer.addChannel(channel)
        deep.addLayer(layer)
        deep.addInput(input)
        //        let validationError = deep.validateNetwork()
        //  Test for constant
        let outputs : [Float] = [0.5]
        for i in 0..<1000 {
            let inputs = [(Float(arc4random()) / Float(UInt32.max))]
            deep.setInputValues("X", values: inputs)
            deep.feedForward()
            deep.startBatch()
            deep.backPropagate(outputs)
            deep.updateWeights((Float(i) / 1000.0) + 0.1, weightDecay: 1.0)
        }
        var totalError : Float = 0.0
        for _ in 0..<100 {
            let inputs = [(Float(arc4random()) / Float(UInt32.max))]
            deep.setInputValues("X", values: inputs)
            deep.feedForward()
            totalError += deep.getTotalError(outputs)
        }
        XCTAssert(totalError < 0.01, "DeepNetwork single NN node constant error")
        
        //  Try a constant slope
        deep.initializeParameters()
        for i in 0..<1000 {
            let inputs = [(Float(arc4random()) / (Float(UInt32.max) * 2.0))+0.25]
            let outputs : [Float] = [inputs[0]*0.5]
            deep.setInputValues("X", values: inputs)
            deep.feedForward()
            deep.startBatch()
            deep.backPropagate(outputs)
            deep.updateWeights((Float(i) / 1000.0) + 0.1, weightDecay: 1.0)
        }
        totalError = 0.0
        for _ in 0..<100 {
            let inputs = [(Float(arc4random()) / (Float(UInt32.max) * 2.0))+0.25]
            let outputs : [Float] = [inputs[0]*0.5]
            deep.setInputValues("X", values: inputs)
            deep.feedForward()
            totalError += deep.getTotalError(outputs)
        }
        XCTAssert(totalError < 0.01, "DeepNetwork single NN node constant slope error")
    }
    
    func testThreeOperatorNN_XOR() {
        let deep = DeepNetwork()
        let layer = DeepLayer()
        let channel = DeepChannel(identifier: "X", sourceChannels: ["X"])
        let op1 = DeepNeuralNetwork(activation: .hyperbolicTangent, size: DeepChannelSize(dimensionCount: 1, dimensionValues: [2]))
        let op2 = DeepNeuralNetwork(activation: .hyperbolicTangent, size: DeepChannelSize(dimensionCount: 1, dimensionValues: [2]))
        let op3 = DeepNeuralNetwork(activation: .hyperbolicTangent, size: DeepChannelSize(dimensionCount: 1, dimensionValues: [1]))
        let input = DeepNetworkInput(inputID: "X", size: DeepChannelSize(dimensionCount: 1, dimensionValues: [2]), values: [0.0, 0.0])
        channel.addNetworkOperator(op1)
        channel.addNetworkOperator(op2)
        channel.addNetworkOperator(op3)
        layer.addChannel(channel)
        deep.addLayer(layer)
        deep.addInput(input)
        //        let validationError = deep.validateNetwork()
        //  Test for XOR
        let inputs : [[Float]] = [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]
        let outputs : [Int] = [0, 1, 1, 0]
        for i in 0..<2000 {
            let inputIndex = Int(arc4random_uniform(4))
            deep.setInputValues("X", values: inputs[inputIndex])
            deep.feedForward()
            deep.startBatch()
            deep.backPropagate(outputs[inputIndex])
            deep.updateWeights((Float(i) / 2000.0) + 0.1, weightDecay: 1.0)
        }
        var totalError : Int = 0
        for i in 0..<4 {
            deep.setInputValues("X", values: inputs[i])
            deep.feedForward()
            if (deep.getResultClass() != outputs[i]) { totalError += 1 }
        }
        XCTAssert(totalError == 0, "DeepNetwork XOR error")
    }
    
    func testSinglePoolN() {
        let deep = DeepNetwork()
        let layer = DeepLayer()
        let channel = DeepChannel(identifier: "X", sourceChannels: ["X"])
        let op = Pooling(type: .maximum, reduction: [2, 2])
        let input = DeepNetworkInput(inputID: "X", size: DeepChannelSize(dimensionCount: 2, dimensionValues: [4, 4]), values:
                    [1.0, 2.0, 1.0, 2.0,
                     3.0, 4.0, 3.0, 1.0,
                     1.0, 2.0, 4.0, 3.0,
                     2.0, 1.0, 1.0, 2.0])
        channel.addNetworkOperator(op)
        layer.addChannel(channel)
        deep.addLayer(layer)
        deep.addInput(input)
        //        let validationError = deep.validateNetwork()
        let results = deep.feedForward()
        XCTAssert(results.count == 4, "DeepNetwork Pooling result count error")
        XCTAssert(results[0] == 4.0, "DeepNetwork Pooling result 0 error")
        XCTAssert(results[1] == 3.0, "DeepNetwork Pooling result 1 error")
        XCTAssert(results[2] == 2.0, "DeepNetwork Pooling result 2 error")
        XCTAssert(results[3] == 4.0, "DeepNetwork Pooling result 3 error")
        
        let gradient = op.backPropogateGradient([1.0, 2.0, 3.0, 4.0])
        XCTAssert(gradient.count == 16, "DeepNetwork Pooling gradient count error")
        XCTAssert(gradient[0] == 0.0, "DeepNetwork Pooling gradient 0 error")
        XCTAssert(gradient[1] == 0.0, "DeepNetwork Pooling gradient 1 error")
        XCTAssert(gradient[2] == 0.0, "DeepNetwork Pooling gradient 2 error")
        XCTAssert(gradient[3] == 0.0, "DeepNetwork Pooling gradient 3 error")
        XCTAssert(gradient[4] == 0.0, "DeepNetwork Pooling gradient 4 error")
        XCTAssert(gradient[5] == 1.0, "DeepNetwork Pooling gradient 5 error")
        XCTAssert(gradient[6] == 2.0, "DeepNetwork Pooling gradient 6 error")
        XCTAssert(gradient[7] == 0.0, "DeepNetwork Pooling gradient 7 error")
        XCTAssert(gradient[8] == 0.0, "DeepNetwork Pooling gradient 8 error")
        XCTAssert(gradient[9] == 3.0, "DeepNetwork Pooling gradient 9 error")
        XCTAssert(gradient[10] == 4.0, "DeepNetwork Pooling gradient 10 error")
        XCTAssert(gradient[11] == 0.0, "DeepNetwork Pooling gradient 11 error")
        XCTAssert(gradient[12] == 0.0, "DeepNetwork Pooling gradient 12 error")
        XCTAssert(gradient[13] == 0.0, "DeepNetwork Pooling gradient 13 error")
        XCTAssert(gradient[14] == 0.0, "DeepNetwork Pooling gradient 14 error")
        XCTAssert(gradient[15] == 0.0, "DeepNetwork Pooling gradient 15 error")
    }
    
    func testSingleConvolutionN() {
        let deep = DeepNetwork()
        let layer = DeepLayer()
        let channel = DeepChannel(identifier: "X", sourceChannels: ["X"])
        let op = Convolution2D(usingMatrix: Convolution2DMatrix.verticalEdge3)      // matrix [-1, 0, 1, -2, 0, 2, -1, 0 , 1]
        let input = DeepNetworkInput(inputID: "X", size: DeepChannelSize(dimensionCount: 2, dimensionValues: [4, 4]), values:
            [1.0, 0.0, 1.0, 0.0,
             1.0, 0.0, 1.0, 0.0,
             1.0, 0.0, 1.0, 0.0,
             1.0, 0.0, 1.0, 0.0])
        channel.addNetworkOperator(op)
        layer.addChannel(channel)
        deep.addLayer(layer)
        deep.addInput(input)
        //        let validationError = deep.validateNetwork()
        let results = deep.feedForward()
        XCTAssert(results.count == 16, "DeepNetwork Convolution result count error")
        XCTAssert(results[0] == -4.0, "DeepNetwork Convolution result 0 error")
        XCTAssert(results[1] ==  0.0, "DeepNetwork Convolution result 1 error")
        XCTAssert(results[2] ==  0.0, "DeepNetwork Convolution result 2 error")
        XCTAssert(results[3] == -4.0, "DeepNetwork Convolution result 3 error")
        XCTAssert(results[4] == -4.0, "DeepNetwork Convolution result 4 error")
        XCTAssert(results[5] ==  0.0, "DeepNetwork Convolution result 5 error")
        XCTAssert(results[6] ==  0.0, "DeepNetwork Convolution result 6 error")
        XCTAssert(results[7] == -4.0, "DeepNetwork Convolution result 7 error")
        XCTAssert(results[8] == -4.0, "DeepNetwork Convolution result 8 error")
        XCTAssert(results[9] ==  0.0, "DeepNetwork Convolution result 9 error")
        XCTAssert(results[10] ==  0.0, "DeepNetwork Convolution result 10 error")
        XCTAssert(results[11] == -4.0, "DeepNetwork Convolution result 11 error")
        XCTAssert(results[12] == -4.0, "DeepNetwork Convolution result 12 error")
        XCTAssert(results[13] ==  0.0, "DeepNetwork Convolution result 13 error")
        XCTAssert(results[14] ==  0.0, "DeepNetwork Convolution result 14 error")
        XCTAssert(results[15] == -4.0, "DeepNetwork Convolution result 15 error")
        
        let gradient = op.backPropogateGradient([1.0, 2.0, 3.0, 4.0, 11.0, 12.0, 13.0, 14.0, 21.0, 22.0, 23.0, 24.0, 31.0, 32.0, 33.0, 34.0])
        XCTAssert(gradient.count == 16, "DeepNetwork Convolution gradient count error")
        XCTAssert(gradient[0] == 1.0, "DeepNetwork Convolution gradient 0 error")
        XCTAssert(gradient[1] == 2.0, "DeepNetwork Convolution gradient 1 error")
        XCTAssert(gradient[2] == 3.0, "DeepNetwork Convolution gradient 2 error")
        XCTAssert(gradient[3] == 4.0, "DeepNetwork Convolution gradient 3 error")
        XCTAssert(gradient[4] == 11.0, "DeepNetwork Convolution gradient 4 error")
        XCTAssert(gradient[5] == 12.0, "DeepNetwork Convolution gradient 5 error")
        XCTAssert(gradient[6] == 13.0, "DeepNetwork Convolution gradient 6 error")
        XCTAssert(gradient[7] == 14.0, "DeepNetwork Convolution gradient 7 error")
        XCTAssert(gradient[8] == 21.0, "DeepNetwork Convolution gradient 8 error")
        XCTAssert(gradient[9] == 22.0, "DeepNetwork Convolution gradient 9 error")
        XCTAssert(gradient[10] == 23.0, "DeepNetwork Convolution gradient 10 error")
        XCTAssert(gradient[11] == 24.0, "DeepNetwork Convolution gradient 11 error")
        XCTAssert(gradient[12] == 31.0, "DeepNetwork Convolution gradient 12 error")
        XCTAssert(gradient[13] == 32.0, "DeepNetwork Convolution gradient 13 error")
        XCTAssert(gradient[14] == 33.0, "DeepNetwork Convolution gradient 14 error")
        XCTAssert(gradient[15] == 34.0, "DeepNetwork Convolution gradient 15 error")
    }

    func testPerformanceExample() {
        // This is an example of a performance test case.
        self.measure {
            // Put the code you want to measure the time of here.
        }
    }

}
