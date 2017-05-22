//
//  PCATests.swift
//  AIToolbox
//
//  Created by Kevin Coble on 3/24/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import XCTest
import AIToolbox

class PCATests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }

    func testExample() {
        //  Create a random 3-D vector for the actual basis
        var initBasis : [Double] = []
        while (true) {
            initBasis = [Double(arc4random()) / Double(UInt32.max), Double(arc4random()) / Double(UInt32.max), Double(arc4random()) / Double(UInt32.max)]
            if initBasis[0] > 0.0 { break } //  Make sure first axis is non zero to make testing easy
        }
        
        //  Create a dataset that is on the basis vector, with a small amount of noise
        let data = DataSet(dataType: .regression, inputDimension: 3, outputDimension: 1)
        var vector = [0.0, 0.0, 0.0]
        do {
            for _ in 0..<20 {
                let scale = Double(arc4random()) * 200.0  / Double(UInt32.max) - 100.0
                vector[0] = scale * initBasis[0] + Gaussian.gaussianRandom(0.0, standardDeviation: 0.01)
                vector[1] = scale * initBasis[1] + Gaussian.gaussianRandom(0.0, standardDeviation: 0.01)
                vector[2] = scale * initBasis[2] + Gaussian.gaussianRandom(0.0, standardDeviation: 0.01)
                try data.addUnlabeledDataPoint(input: vector)
            }
        }
        catch {
            print("Invalid data set created")
        }
        
        //  Create a PCA class and do the PCA reduction of the basis vectors
        let pca = PCA(initialSize: 3, reduceSize: 1)
        do {
            try pca.getReducedBasisVectorSet(data)
        }
        catch {
            print("Error getting reduced basis vector set")
        }
        
        //  Get the ratio of the basis vector to the initBasis vector
        let ratio = pca.basisVectors[0] / initBasis[0]
        
        //  Compare the second axis
        var difference = ratio * initBasis[1] - pca.basisVectors[1]
        XCTAssert(fabs(difference) < 0.02, "PCA Y axis")
        
        //  Compare the third axis
        difference = ratio * initBasis[2] - pca.basisVectors[2]
        XCTAssert(fabs(difference) < 0.02, "PCA Z axis")
        
        //  Create a transform test data set that is a known distance along the new basis vector
        let testData = DataSet(dataType: .regression, inputDimension: 3, outputDimension: 1)
        do {
            vector[0] = pca.basisVectors[0] * 5.0 + pca.μ[0]
            vector[1] = pca.basisVectors[1] * 5.0 + pca.μ[1]
            vector[2] = pca.basisVectors[2] * 5.0 + pca.μ[2]
            try testData.addUnlabeledDataPoint(input: vector)
        }
        catch {
            print("Invalid data set created")
        }
        
        //  Transform the data set
        do {
            let transformedData = try pca.transformDataSet(testData)
            let range = (transformedData as MLRegressionDataSet).getInputRange()
            XCTAssert(range.count == 1, "PCA transformed data size")
            let transformedVector = try transformedData.getInput(0)
            XCTAssert(fabs(transformedVector[0] - 5) < 0.1, "PCA transformed data")     //  input vector five times basis vector, result should be a 5
        }
        catch {
            print("Error transforming data set")
        }
        //  Test persistence routines
//                let path = "SomeValidWritablePath"
//                do {
//                    try pca.saveToFile(path)
//                }
//                catch {
//                    print("Error in writing file")
//                }
//                let readPCA = PCA(loadFromFile: path)
//                if let readPCA = readPCA {
//                    XCTAssert(pca.initialDimension == readPCA.initialDimension, "PCA persistance data")
//                    XCTAssert(pca.reducedDimension == readPCA.reducedDimension, "PCA persistance data")
//                    XCTAssert(pca.μ[0] == readPCA.μ[0], "PCA persistance data")
//                    XCTAssert(pca.eigenValues[0] == readPCA.eigenValues[0], "PCA persistance data")
//                    XCTAssert(pca.basisVectors[0] == readPCA.basisVectors[0], "PCA persistance data")
//                }
    }
    
    func testExample2() {
        //  Create a dataset that is on the X and Z vectors, with a small amount of noise
        let data = DataSet(dataType: .regression, inputDimension: 3, outputDimension: 1)
        var vector = [0.0, 0.0, 0.0]
        do {
            for _ in 0..<20 {
                let scale = Double(arc4random()) * 200.0  / Double(UInt32.max) - 100.0
                if (arc4random() % 2 == 0) {
                    //  Along the X axis
                    vector[0] = scale + Gaussian.gaussianRandom(0.0, standardDeviation: 0.01)
                    vector[2] = Gaussian.gaussianRandom(0.0, standardDeviation: 0.01)
                }
                else {
                    //  Along the Z axis
                    vector[2] = scale + Gaussian.gaussianRandom(0.0, standardDeviation: 0.01)
                    vector[0] = Gaussian.gaussianRandom(0.0, standardDeviation: 0.01)
                }
                try data.addUnlabeledDataPoint(input: vector)
            }
        }
        catch {
            print("Invalid data set created")
        }
        
        //  Create a PCA class and do the PCA reduction of the basis vectors
        let pca = PCA(initialSize: 3, reduceSize: 2)
        do {
            try pca.getReducedBasisVectorSet(data)
        }
        catch {
            print("Error getting reduced basis vector set")
        }
        
        //  Verify the Y dimension is no longer relevant
        XCTAssert(fabs(pca.basisVectors[1]) < 0.02, "PCA Y axis irrelevant")
        XCTAssert(fabs(pca.basisVectors[4]) < 0.02, "PCA Y axis irrelevant")
        
        //  Create a transform test data set that is a known distance along the X axis, and has a Y component
        let testData = DataSet(dataType: .regression, inputDimension: 3, outputDimension: 1)
        do {
            vector[0] = pca.basisVectors[0] * 10.0 + pca.μ[0]
            vector[1] = 1.0 + pca.μ[1]
            vector[2] = pca.μ[2]
            try testData.addUnlabeledDataPoint(input: vector)
        }
        catch {
            print("Invalid data set created")
        }
        
        //  Transform the data set
        do {
            let transformedData = try pca.transformDataSet(testData)
            let transformedVector = try transformedData.getInput(0)
            XCTAssert(transformedVector.count == 2, "PCA transformed data size")
        }
        catch {
            print("Error transforming data set")
        }
        //  Test persistence routines
        //                let path = "SomeValidWritablePath"
        //                do {
        //                    try pca.saveToFile(path)
        //                }
        //                catch {
        //                    print("Error in writing file")
        //                }
        //                let readPCA = PCA(loadFromFile: path)
        //                if let readPCA = readPCA {
        //                    XCTAssert(pca.initialDimension == readPCA.initialDimension, "PCA persistance data")
        //                    XCTAssert(pca.reducedDimension == readPCA.reducedDimension, "PCA persistance data")
        //                    XCTAssert(pca.μ[0] == readPCA.μ[0], "PCA persistance data")
        //                    XCTAssert(pca.eigenValues[0] == readPCA.eigenValues[0], "PCA persistance data")
        //                    XCTAssert(pca.basisVectors[0] == readPCA.basisVectors[0], "PCA persistance data")
        //                }
    }

    func testPerformanceExample() {
        // This is an example of a performance test case.
        self.measure {
            // Put the code you want to measure the time of here.
        }
    }

}
