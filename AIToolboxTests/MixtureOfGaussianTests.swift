//
//  MixtureOfGaussianTests.swift
//  AIToolbox
//
//  Created by Kevin Coble on 4/13/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import XCTest
import AIToolbox

class MixtureOfGaussianTests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }

    func testTwo() {
        //  Create two gaussians at random locations to make a test data set
        let gauss1x = Double(arc4random()) * 100.0 / Double(UInt32.max)
        let gauss1y = Double(arc4random()) * 100.0 / Double(UInt32.max)
        let gauss2x = Double(arc4random()) * 100.0 / Double(UInt32.max)
        let gauss2y = Double(arc4random()) * 100.0 / Double(UInt32.max)
        let gauss1α = Double(arc4random()) / Double(UInt32.max)
        var gauss1 : MultivariateGaussian?
        var gauss2 : MultivariateGaussian?
        do {
            gauss1 = try MultivariateGaussian(dimension: 2)
            gauss2 = try MultivariateGaussian(dimension: 2)
            try gauss1!.setMean([gauss1x, gauss1y])
            try gauss2!.setMean([gauss2x, gauss2y])
        }
        catch {
            print("error creating gaussian")
            return
        }
        
        //  Create a dataset to train on
        let trainData = DataSet(dataType: .regression, inputDimension: 2, outputDimension: 1)
        do {
            for _ in 0..<500 {
                let x = Double(arc4random()) * 100.0 / Double(UInt32.max)
                let y = Double(arc4random()) * 100.0 / Double(UInt32.max)
                let input = [x, y]
                let source = Double(arc4random()) / Double(UInt32.max)
                if (source < gauss1α) {
                    let result = try gauss1!.getProbability(input)
                    try trainData.addDataPoint(input: input, output: [result])
                }
                else {
                    let result = try gauss2!.getProbability(input)
                    try trainData.addDataPoint(input: input, output: [result])
                }
            }
        }
        catch {
            print("error creating test data")
        }
        
        //  Create a dataset to test on
        let testData = DataSet(dataType: .regression, inputDimension: 2, outputDimension: 1)
        do {
            for _ in 0..<100 {
                let x = Double(arc4random()) * 100.0 / Double(UInt32.max)
                let y = Double(arc4random()) * 100.0 / Double(UInt32.max)
                let input = [x, y]
                let source = Double(arc4random()) / Double(UInt32.max)
                if (source < gauss1α) {
                    let result = try gauss1!.getProbability(input)
                    try testData.addDataPoint(input: input, output: [result])
                }
                else {
                    let result = try gauss2!.getProbability(input)
                    try testData.addDataPoint(input: input, output: [result])
                }
            }
        }
        catch {
            print("error creating test data")
        }
        
        //  Create and train a 'mixture of gaussians'
        do {
            let mog = try MixtureOfGaussians(inputSize: 2, numberOfTerms: 2, diagonalCoVariance: true)
            mog.convergenceLimit = 0.000000001
            try mog.trainRegressor(trainData)
            
            //  See if it predicted the data source setup
            let averageError = try mog.getTotalAbsError(testData) / Double(trainData.size)
            XCTAssert(averageError < 0.01, "mixture-of-gaussians validation error")
        }
        catch {
            print("error training mixture of gaussians")
        }
    }

    func testPerformanceExample() {
        // This is an example of a performance test case.
        self.measure {
            // Put the code you want to measure the time of here.
        }
    }

}
