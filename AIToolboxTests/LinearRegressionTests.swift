//
//  LinearRegressionTests.swift
//  AIToolbox
//
//  Created by Kevin Coble on 4/4/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import XCTest
import AIToolbox

class LinearRegressionTests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }

    //  Test standard Ax + B
    func testLine() {
        //  Create a data set
        let slope = Double(arc4random()) / Double(UInt32.max)
        let offset = Double(arc4random()) * 100.0 / Double(UInt32.max)
        let data = DataSet(dataType: .regression, inputDimension: 1, outputDimension: 1)
        var inputs = [0.0]
        do {
            for _ in 0..<50 {
                inputs[0] = Double(arc4random()) * 100.0 / Double(UInt32.max)
                let output = inputs[0] * slope + offset + Gaussian.gaussianRandom(0.0, standardDeviation: 0.01)
                try data.addDataPoint(input: inputs, output: [output])
            }
        }
        catch {
            print("Invalid data set created")
        }
        
        //  Create a model
        let lr = LinearRegressionModel(inputSize: 1, outputSize: 1, polygonOrder: 1)
        
        //  Train the model
        do {
            try lr.trainRegressor(data)
        }
        catch {
            print("Linear Regression Training error")
        }
        
        //  Check the results
        var difference = lr.Θ[0][0] - offset
        XCTAssert(fabs(difference) < 0.02, "Linear Regression Line offset")
        difference = lr.Θ[0][1] - slope
        XCTAssert(fabs(difference) < 0.02, "Linear Regression Line slope")
        inputs[0] = Double(arc4random()) * 100.0 / Double(UInt32.max)
        let expectedOutput = inputs[0] * slope + offset
        do {
            let results = try lr.predictOne(inputs)
            difference = fabs(expectedOutput - results[0])
            XCTAssert(difference < 0.01, "Linear Regression predict")
        }
        catch {
            print("Linear Regression Training error")
        }
        //  Create a data set of 4 training examples
        let testData = DataSet(dataType: .regression, inputDimension: 1, outputDimension: 1)
        var expectedOutputs : [Double] = []
        do {
            for _ in 0..<4 {
                let input = Double(arc4random()) * 100.0 / Double(UInt32.max)
                try testData.addTestDataPoint(input: [input])
                expectedOutputs.append(input * slope + offset)
            }
        }
        catch {
            print("Invalid data set created")
        }
        //  Process the data
        do {
            try lr.predict(testData)
            for index in 0..<4 {
                let difference = fabs(expectedOutputs[index] - testData.singleOutput(index)!)
                XCTAssert(difference < 0.01, "Linear Regression predict")
            }
        }
        catch {
            print("Error in Linear Regression prediction")
        }
    }
    
    func testRegularization() {
        //  Create a data set
        let offset = Double(arc4random()) / Double(UInt32.max)
        let slope = Double(arc4random()) / Double(UInt32.max)
        let roc = Double(arc4random()) / Double(UInt32.max)
        let rroc = Double(arc4random()) / Double(UInt32.max)
        let data = DataSet(dataType: .regression, inputDimension: 1, outputDimension: 1)
        var inputs = [0.0]
        do {
            for _ in 0..<50 {
                inputs[0] = Double(arc4random()) * 100.0 / Double(UInt32.max)
                let output = (inputs[0] * inputs[0] * inputs[0] * rroc) + (inputs[0] * inputs[0] * roc) + (inputs[0] * slope) + offset + (Double(arc4random()) * 0.1 / Double(UInt32.max)) - 0.05
                try data.addDataPoint(input: inputs, output: [output])
            }
        }
        catch {
            print("Invalid data set created")
        }
        
        //  Create a model
        let lr = LinearRegressionModel(inputSize: 1, outputSize: 1, polygonOrder: 3)
        lr.regularization = 100.0     //  High enough to lower the parameters well
        
        //  Train the model
        do {
            try lr.trainRegressor(data)
        }
        catch {
            print("Linear Regression Training error")
        }
        
        //  Check the results
        let difference = (offset + slope + roc + rroc) - (lr.Θ[0][0] + lr.Θ[0][1] + lr.Θ[0][2] + lr.Θ[0][3])    //  Verify regularization lowered it
        XCTAssert(difference > 0.2, "Linear Regression regularization")
    }

    func testPerformanceExample() {
        // This is an example of a performance test case.
        self.measure {
            // Put the code you want to measure the time of here.
        }
    }

}
