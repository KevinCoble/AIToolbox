//
//  GaussianTests.swift
//  AIToolbox
//
//  Created by Kevin Coble on 4/11/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import XCTest
import AIToolbox

class GaussianTests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testGaussian() {
        do {
            let g = try Gaussian(mean: 1.0, variance: 1.0)
            var probability = g.getProbability(1.0)
            XCTAssert(fabs(probability - 0.39894228040143) < 0.000001, "single variable gaussian probability")
            probability = g.getProbability(2.0)
            XCTAssert(fabs(probability - 0.24197072) < 0.000001, "single variable gaussian probability")
        }
        catch {
            print("error creating gaussian")
        }
    }

    func testMultiVariate() {
        //  Create a multivariate gaussian
        do {
            let gaussian = try MultivariateGaussian(dimension: 2, diagonalCovariance: false)
            try gaussian.setCoVariance(0, inputIndex2: 0, value: 2.0)
            try gaussian.setCoVariance(1, inputIndex2: 1, value: 3.0)
            try gaussian.setCoVariance(0, inputIndex2: 1, value: 1.0)
            
            let probability = try gaussian.getProbability([1.0, 1.0])
            XCTAssert(fabs(probability - 0.052728666096220705) < 0.000001, "multi variable gaussian probability")
        }
        catch {
            print("error creating gaussian")
        }
    }

    func testPerformanceExample() {
        // This is an example of a performance test case.
        self.measure {
            // Put the code you want to measure the time of here.
        }
    }

}
