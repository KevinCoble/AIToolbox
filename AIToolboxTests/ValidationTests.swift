//
//  ValidationTests.swift
//  AIToolbox
//
//  Created by Kevin Coble on 4/24/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import XCTest
import AIToolbox

class ValidationTests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }

    func testSimpleValidation() {
        //  Create a data set, using 2x² + 3x + 4 as the target function
        let data = DataSet(dataType : .regression, inputDimension : 1, outputDimension : 1)
        do {
            for _ in 0..<500 {
                let x = Double(arc4random()) * 200.0 / Double(UInt32.max) - 100.0
                let y = 2.0 * x * x + 3.0 * x + 4 + Gaussian.gaussianRandom(0.0, standardDeviation: 0.1)
                try data.addDataPoint(input: [x], output: [y])
            }
        }
        catch {
            print("error creating test data")
        }
        
        //  Create a validation set, with three different linear regression models
        let validation = Validation(type: .regression)
        let line = LinearRegressionModel(inputSize: 1, outputSize: 1, polygonOrder: 1)          //   Ax + B
        let quadratic = LinearRegressionModel(inputSize: 1, outputSize: 1, polygonOrder: 2)     //   Ax² + Bx + C
        let exponential = LinearRegressionModel(inputSize: 1, outputSize: 1)                    //   Ae^x
        var exponentialSubTerm = LinearRegressionSubTerm(withInput: 0)
        exponentialSubTerm.function = SubtermFunction.naturalExponent
        var exponentialTerm = LinearRegressionTerm()
        exponentialTerm.addSubTerm(exponentialSubTerm)
        exponential.addTerm(exponentialTerm)
        do {
            try validation.addModel(line)
            try validation.addModel(quadratic)
            try validation.addModel(exponential)
        }
        catch {
            print("error creating validation object")
        }
        
        //  Validate the models with 20% of the data
        do {
            let bestModel = try validation.simpleValidation(data, fractionForTest: 0.2)
            XCTAssert(bestModel == 1, "Simple Validation")
        }
        catch {
            print("error validation models")
        }
    }
    
    func testNFoldValidation() {
        //  Create a data set, using 2x² + 3x + 4 as the target function
        let data = DataSet(dataType : .regression, inputDimension : 1, outputDimension : 1)
        do {
            for i in 0..<5 {
                let x =  Double(i) * 10
                let y = 2.0 * x * x + 3.0 * x + 4 + Gaussian.gaussianRandom(0.0, standardDeviation: 0.1)
                try data.addDataPoint(input: [x], output: [y])
            }
        }
        catch {
            print("error creating test data")
        }
        
        //  Create a validation set, with three different linear regression models
        let validation = Validation(type: .regression)
        let line = LinearRegressionModel(inputSize: 1, outputSize: 1, polygonOrder: 1)          //   Ax + B
        let quadratic = LinearRegressionModel(inputSize: 1, outputSize: 1, polygonOrder: 2)     //   Ax² + Bx + C
        let exponential = LinearRegressionModel(inputSize: 1, outputSize: 1)                    //   Ae^x
        var exponentialSubTerm = LinearRegressionSubTerm(withInput: 0)
        exponentialSubTerm.function = SubtermFunction.naturalExponent
        var exponentialTerm = LinearRegressionTerm()
        exponentialTerm.addSubTerm(exponentialSubTerm)
        exponential.addTerm(exponentialTerm)
        do {
            try validation.addModel(line)
            try validation.addModel(quadratic)
            try validation.addModel(exponential)
        }
        catch {
            print("error creating validation object")
        }
        
        //  Validate the models with 5 sections of the data
        do {
            let bestModel = try validation.NFoldCrossValidation(data, numberOfFolds: data.size)
            XCTAssert(bestModel == 1, "N-Fold validation")
        }
        catch {
            print("error validation models")
        }
    }

    func testPerformanceExample() {
        // This is an example of a performance test case.
        self.measure {
            // Put the code you want to measure the time of here.
        }
    }

}
