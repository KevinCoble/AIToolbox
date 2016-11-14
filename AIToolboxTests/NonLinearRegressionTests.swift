//
//  NonLinearRegressionTests.swift
//  AIToolbox
//
//  Created by Kevin Coble on 4/15/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation
import XCTest
import AIToolbox

class NonLinearRegressionTests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testParameterDelta() {
        //  Create the model equation
        let equation = SimpleEquation()
        
        //  Set it with random parameters
        equation.parameters = [Double(arc4random()) * 100.0 / Double(UInt32.max) - 50.0, Double(arc4random()) * 4.0 / Double(UInt32.max) - 2.0]
        
        //  Create a dataset to train on
        let trainData = DataSet(dataType: .regression, inputDimension: 1, outputDimension: 1)
        do {
            for _ in 0..<1000 {
                let x = [Double(arc4random()) * 10.0 / Double(UInt32.max) - 5.0]
                let y = try equation.getOutputs(x)
                try trainData.addDataPoint(input: x, output: y)
            }
        }
        catch {
            print("error creating test data")
        }
        
        //  Create a dataset to test on
        let testData = DataSet(dataType: .regression, inputDimension: 1, outputDimension: 1)
        do {
            for _ in 0..<100 {
                let x = [Double(arc4random()) * 10.0 / Double(UInt32.max) - 5.0]
                let y = try equation.getOutputs(x)
                try testData.addDataPoint(input: x, output: y)
            }
        }
        catch {
            print("error creating test data")
        }
        
        //  Create the ParameterDelta non-linear regressor
        let regressor = NonLinearRegression(equation: equation, batchSize: 100, initialDelta: 0.1)
        
        //  Train the model
        do {
            try regressor.trainRegressor(trainData)
        }
        catch {
            print("Non-Linear Regression Training error")
        }
        
        //  See if it predicted the data source setup
        do {
            let averageError = try regressor.getTotalAbsError(testData) / Double(testData.size)
            XCTAssert(averageError < 0.05, "non-linear ParameterDelta validation error at \(averageError)")
        }
        catch {
            print("Non-Linear Regression Testing error")
        }
    }

    func testSGD() {
        //  Create the model equation
        let equation = SimpleEquation()
        
        //  Set it with random parameters
        equation.parameters = [Double(arc4random()) * 100.0 / Double(UInt32.max) - 50.0, Double(arc4random()) * 4.0 / Double(UInt32.max) - 2.0]
        
        //  Create a dataset to train on
        let trainData = DataSet(dataType: .regression, inputDimension: 1, outputDimension: 1)
        do {
            for _ in 0..<1000 {
                let x = [Double(arc4random()) * 10.0 / Double(UInt32.max) - 5.0]
                let y = try equation.getOutputs(x)
                try trainData.addDataPoint(input: x, output: y)
            }
        }
        catch {
            print("error creating test data")
        }
        
        //  Create a dataset to test on
        let testData = DataSet(dataType: .regression, inputDimension: 1, outputDimension: 1)
        do {
            for _ in 0..<100 {
                let x = [Double(arc4random()) * 10.0 / Double(UInt32.max) - 5.0]
                let y = try equation.getOutputs(x)
                try testData.addDataPoint(input: x, output: y)
            }
        }
        catch {
            print("error creating test data")
        }
        
        //  Create the SGD non-linear regressor
        let regressor = NonLinearRegression(equation: equation, batchSize: 100, initialStepSize: 0.1, multiplyBy: 1.5, afterIterations: 100)
        regressor.setConvergence(.smallAverageLoss, limit: 0.001)
        
        //  Train the model
        do {
            try regressor.trainRegressor(trainData)
        }
        catch {
            print("Non-Linear Regression Training error")
        }
        
        //  See if it predicted the data source setup
        do {
            let averageError = try regressor.getTotalAbsError(testData) / Double(testData.size)
            XCTAssert(averageError < 0.05, "non-linear SGD validation error at \(averageError)")
        }
        catch {
            print("Non-Linear Regression Testing error")
        }
    }
    
    func testGaussNewton() {
        //  Create the model equation
        let equation = ExponentEquation()
        
        //  Create a dataset to train on (From "Solving NonLinear Least-Squares Problems with the Gauss-Newton and Levenberg-Marguardt Methods" by Alfonso CroEze, Lindsey Pittmand and Winnie Reynolds)
        let trainData = DataSet(dataType: .regression, inputDimension: 1, outputDimension: 1)
        do {
            try trainData.addDataPoint(input: [1.0], output: [8.3])
            try trainData.addDataPoint(input: [2.0], output: [11.0])
            try trainData.addDataPoint(input: [3.0], output: [14.7])
            try trainData.addDataPoint(input: [4.0], output: [19.7])
            try trainData.addDataPoint(input: [5.0], output: [26.7])
            try trainData.addDataPoint(input: [6.0], output: [35.2])
            try trainData.addDataPoint(input: [7.0], output: [44.4])
            try trainData.addDataPoint(input: [8.0], output: [55.9])
        }
        catch {
            print("error creating training data")
        }
        
        //  Create the SGD non-linear regressor
        let regressor = NonLinearRegression(equation: equation, batchSize: trainData.size)
        regressor.setConvergence(.smallParameterChange, limit: 0.0001)
        
        //  Set an initializer function for the parameters (if not set, random parameters will be used)
        regressor.setCustomInitializer(GNInitializer)
        
        //  Train the model
        do {
            try regressor.trainRegressor(trainData)
        }
        catch {
            print("Non-Linear Regression Training error")
        }
    }
    func GNInitializer(_ trainData: MLDataSet) -> [Double] {
        return [6.0, 0.3]
    }

    func testPerformanceExample() {
        // This is an example of a performance test case.
        self.measure {
            // Put the code you want to measure the time of here.
        }
    }

}

//  Class for non-linear function -> exponent Ae^Bx (two parameters, one input, one output)
open class ExponentEquation : NonLinearEquation {
    open var parameters: [Double] = []
    
    open func getInputDimension() -> Int
    {
        return 1
    }
    open func getOutputDimension() -> Int
    {
        return 1
    }
    open func getParameterDimension() -> Int
    {
        return 2
    }
    open func setParameters(_ parameters: [Double]) throws
    {
        if (parameters.count < getParameterDimension()) { throw MachineLearningError.notEnoughData }
        self.parameters = parameters
    }
    open func getOutputs(_ inputs: [Double]) throws -> [Double]        //  Returns vector outputs sized for outputs
    {
        return [parameters[0] * exp(parameters[1] * inputs[0])]
    }
    open func getGradient(_ inputs: [Double]) throws -> [Double]       //  Returns vector gradient with respect to parameters
    {
        return [exp(parameters[1] * inputs[0]),
                inputs[0] * parameters[0] * exp(parameters[1] * inputs[0])]
    }
}

//  Class for simple function for testing -> A + Bx (two parameters, one input, one output)
open class SimpleEquation : NonLinearEquation {
    open var parameters: [Double] = []
    
    open func getInputDimension() -> Int
    {
        return 1
    }
    open func getOutputDimension() -> Int
    {
        return 1
    }
    open func getParameterDimension() -> Int
    {
        return 2
    }
    open func setParameters(_ parameters: [Double]) throws
    {
        if (parameters.count < getParameterDimension()) { throw MachineLearningError.notEnoughData }
        self.parameters = parameters
    }
    open func getOutputs(_ inputs: [Double]) throws -> [Double]        //  Returns vector outputs sized for outputs
    {
                return [parameters[0] + (parameters[1] * inputs[0])]
    }
    open func getGradient(_ inputs: [Double]) throws -> [Double]       //  Returns vector gradient with respect to parameters
    {
                return [1.0,
                        inputs[0]]
    }
}

