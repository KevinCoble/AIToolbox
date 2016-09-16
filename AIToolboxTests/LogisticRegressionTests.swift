//
//  LogisticRegressionTests.swift
//  AIToolbox
//
//  Created by Kevin Coble on 5/13/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import XCTest
import AIToolbox

class LogisticRegressionTests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }

    //  Test the binary data case
    func testBinary() {
        //  Create test case from Wikipedia article example - https://en.wikipedia.org/wiki/Logistic_regression
        let data = DataSet(dataType: .classification, inputDimension: 1, outputDimension: 1)
        do {
            try data.addDataPoint(input: [0.50], dataClass:0)
            try data.addDataPoint(input: [0.75], dataClass:0)
            try data.addDataPoint(input: [1.00], dataClass:0)
            try data.addDataPoint(input: [1.25], dataClass:0)
            try data.addDataPoint(input: [1.50], dataClass:0)
            try data.addDataPoint(input: [1.75], dataClass:0)
            try data.addDataPoint(input: [1.75], dataClass:1)
            try data.addDataPoint(input: [2.00], dataClass:0)
            try data.addDataPoint(input: [2.25], dataClass:1)
            try data.addDataPoint(input: [2.50], dataClass:0)
            try data.addDataPoint(input: [2.75], dataClass:1)
            try data.addDataPoint(input: [3.00], dataClass:0)
            try data.addDataPoint(input: [3.25], dataClass:1)
            try data.addDataPoint(input: [3.50], dataClass:0)
            try data.addDataPoint(input: [4.00], dataClass:1)
            try data.addDataPoint(input: [4.25], dataClass:1)
            try data.addDataPoint(input: [4.50], dataClass:1)
            try data.addDataPoint(input: [4.75], dataClass:1)
            try data.addDataPoint(input: [5.00], dataClass:1)
            try data.addDataPoint(input: [5.50], dataClass:1)
        }
        catch {
            print("Invalid data set created")
        }
        
        //  Create and train a logistic regression
        let lr = LogisticRegression(numInputs : 1, solvingMethod: .sgd)
        do {
            try lr.trainClassifier(data)
        }
        catch {
            print("Error training logistic regression")
        }
        
        //  Create the test set
        let test = DataSet(dataType: .classification, inputDimension: 1, outputDimension: 1)
        do {
            try test.addTestDataPoint(input: [1.0])         //  Fail
            try test.addTestDataPoint(input: [2.0])         //  Fail
            try test.addTestDataPoint(input: [3.0])         //  Pass
            try test.addTestDataPoint(input: [4.0])         //  Pass
            try test.addTestDataPoint(input: [5.0])         //  Pass
        }
        catch {
            print("Invalid test sequence data set created")
        }
        
        //  Classify the set
        do {
            try lr.classify(test)
        }
        catch {
            print("Error having logistic regression classify")
        }
        
        //  Verify the results
        do {
            var result : Int
            result = try test.getClass(0)
            XCTAssert(result == 0, "logistic regression test 0")
            result = try test.getClass(1)
            XCTAssert(result == 0, "logistic regression test 1")
            result = try test.getClass(2)
            XCTAssert(result == 1, "logistic regression test 2")
            result = try test.getClass(3)
            XCTAssert(result == 1, "logistic regression test 3")
            result = try test.getClass(4)
            XCTAssert(result == 1, "logistic regression test 4")
        }
        catch {
            print("Error getting test results")
        }
    }

    func testPerformanceExample() {
        // This is an example of a performance test case.
        self.measure {
            // Put the code you want to measure the time of here.
        }
    }

}
