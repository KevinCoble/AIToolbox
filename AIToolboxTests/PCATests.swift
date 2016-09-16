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
        
        //  Transorm the data set
        do {
            let transformedData = try pca.transformDataSet(data)
            let range = (transformedData as MLRegressionDataSet).getInputRange()
            XCTAssert(range.count == 1, "PCA transformed data")
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
