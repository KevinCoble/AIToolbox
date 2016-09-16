//
//  KMeansTest.swift
//  AIToolbox
//
//  Created by Kevin Coble on 3/17/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import XCTest
import AIToolbox

class KMeansTest: XCTestCase {

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }

    func testKMeans() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct results.
        //  Create a k-means object for 2 classes
        let kmeans = KMeans(classes: 2)
        
        //  Create a data set of 6 points in two dimension
        let data = DataSet(dataType: .classification, inputDimension: 2, outputDimension: 1)
        do {
            try data.addUnlabeledDataPoint(input: [-1.0, -1.0])
            try data.addUnlabeledDataPoint(input: [1.0, 1.0])
            try data.addUnlabeledDataPoint(input: [-0.9, -1.0])
            try data.addUnlabeledDataPoint(input: [0.9, 1.0])
            try data.addUnlabeledDataPoint(input: [-1.0, -0.9])
            try data.addUnlabeledDataPoint(input: [1.0, 0.9])
        }
        catch {
            print("Invalid data set created")
        }
        
        //  Get the groupings
        do {
            try kmeans.train(data)
        }
        catch {
            print("Error getting kmeans groupings")
        }
        
        //  Test the grouping
        var classLabel : Int
        var classLabel2 : Int
        do {
            try classLabel = data.getClass(0)
            try classLabel2 = data.getClass(2)
            XCTAssert(classLabel == classLabel2, "point 2 did not go into the same group as point 0")
            try classLabel2 = data.getClass(4)
            XCTAssert(classLabel == classLabel2, "point 4 did not go into the same group as point 0")
            try classLabel = data.getClass(1)
            try classLabel2 = data.getClass(3)
            XCTAssert(classLabel == classLabel2, "point 3 did not go into the same group as point 1")
            try classLabel2 = data.getClass(5)
            XCTAssert(classLabel == classLabel2, "point 5 did not go into the same group as point 1")
        }
        catch {
            print("Error checking kmeans groupings")
        }
    }

    func testPerformanceExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct results.
        //  Create a k-means object for 16 classes
        let kmeans = KMeans(classes: 16)
        
        //  Create a data set of 1000 points in four dimension
        let data = DataSet(dataType: .classification, inputDimension: 4, outputDimension: 1)
        do {
            for _ in 0..<1000 {
                var inputs: [Double] = []
                inputs.append(Double(arc4random()) / Double(UInt32.max))
                inputs.append(Double(arc4random()) / Double(UInt32.max))
                inputs.append(Double(arc4random()) / Double(UInt32.max))
                inputs.append(Double(arc4random()) / Double(UInt32.max))
                try data.addUnlabeledDataPoint(input: inputs)
            }
        }
        catch {
            print("Invalid data set created")
        }
        
        self.measure {
            //  Get the groupings
            do {
                try kmeans.train(data)
            }
            catch {
                print("Error getting kmeans groupings")
            }
        }
    }

}
