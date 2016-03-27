//
//  SVMTests.swift
//  AIToolbox
//
//  Created by Kevin Coble on 12/6/15.
//  Copyright © 2015 Kevin Coble. All rights reserved.
//

import XCTest
import AIToolbox

class SVMTests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }

    func testClassification() {
        //  Create a data set
        let data = DataSet(dataType: .Classification, inputDimension: 2, outputDimension: 1)
        do {
            try data.addDataPoint(input: [0.0, 1.0], output:1)
            try data.addDataPoint(input: [0.0, 0.9], output:1)
            try data.addDataPoint(input: [0.1, 1.0], output:1)
            try data.addDataPoint(input: [1.0, 0.0], output:0)
            try data.addDataPoint(input: [1.0, 0.1], output:0)
            try data.addDataPoint(input: [0.9, 0.0], output:0)
        }
        catch {
            print("Invalid data set created")
        }
        
        //  Create an SVM classifier and train
        let svm = SVMModel(problemType: .C_SVM_Classification, kernelSettings:
                    KernelParameters(type: .RadialBasisFunction, degree: 0, gamma: 0.5, coef0: 0.0))
        svm.train(data)
        
        //  Create a test dataset
        let testData = DataSet(dataType: .Classification, inputDimension: 2, outputDimension: 1)
        do {
            try testData.addTestDataPoint(input: [0.0, 0.1])    //  Expect 1
            try testData.addTestDataPoint(input: [0.1, 0.0])    //  Expect 0
            try testData.addTestDataPoint(input: [1.0, 0.9])    //  Expect 0
            try testData.addTestDataPoint(input: [0.9, 1.0])    //  Expect 1
            try testData.addTestDataPoint(input: [0.5, 0.4])    //  Expect 0
            try testData.addTestDataPoint(input: [0.5, 0.6])    //  Expect 1
        }
        catch {
            print("Invalid data set created")
        }
        
        //  Predict on the test data
        svm.predictValues(testData)
        
        //  See if we matched
        var classLabel : Int
        do {
            try classLabel = testData.getClass(0)
            XCTAssert(classLabel == 1, "first test data point, expect 1")
            try classLabel = testData.getClass(1)
            XCTAssert(classLabel == 0, "second test data point, expect 0")
            try classLabel = testData.getClass(2)
            XCTAssert(classLabel == 0, "third test data point, expect 0")
            try classLabel = testData.getClass(3)
            XCTAssert(classLabel == 1, "fourth test data point, expect 1")
            try classLabel = testData.getClass(4)
            XCTAssert(classLabel == 0, "fifth test data point, expect 0")
            try classLabel = testData.getClass(5)
            XCTAssert(classLabel == 1, "sixth test data point, expect 1")
        }
        catch {
            print("Error in prediction")
        }
        
        let path = "/Users/kevincoble/Documents/svm.plist"
        do {
            try svm.saveToFile(path)
        }
        catch {
            print("Error in writing file")
        }
        let readSVM = SVMModel(loadFromFile: path)
        if let readSVM = readSVM {
            readSVM.predictValues(testData)
            
            //  See if we matched
            var classLabel : Int
            do {
                try classLabel = testData.getClass(0)
                XCTAssert(classLabel == 1, "first test data point, expect 1")
                try classLabel = testData.getClass(1)
                XCTAssert(classLabel == 0, "second test data point, expect 0")
                try classLabel = testData.getClass(2)
                XCTAssert(classLabel == 0, "third test data point, expect 0")
                try classLabel = testData.getClass(3)
                XCTAssert(classLabel == 1, "fourth test data point, expect 1")
                try classLabel = testData.getClass(4)
                XCTAssert(classLabel == 0, "fifth test data point, expect 0")
                try classLabel = testData.getClass(5)
                XCTAssert(classLabel == 1, "sixth test data point, expect 1")
            }
            catch {
                print("Error in prediction with read SVM")
            }
        }
    }
    
    func testThreeStateClassification() {
        //  Create a data set
        let data = DataSet(dataType: .Classification, inputDimension: 2, outputDimension: 1)
        do {
            try data.addDataPoint(input: [0.2, 0.9], output:0)
            try data.addDataPoint(input: [0.8, 0.3], output:0)
            try data.addDataPoint(input: [0.5, 0.6], output:0)
            try data.addDataPoint(input: [0.2, 0.7], output:1)
            try data.addDataPoint(input: [0.2, 0.3], output:1)
            try data.addDataPoint(input: [0.4, 0.5], output:1)
            try data.addDataPoint(input: [0.5, 0.4], output:2)
            try data.addDataPoint(input: [0.3, 0.2], output:2)
            try data.addDataPoint(input: [0.7, 0.2], output:2)
        }
        catch {
            print("Invalid data set created")
        }
        
        //  Create an SVM classifier and train
        let svm = SVMModel(problemType: .C_SVM_Classification, kernelSettings:
            KernelParameters(type: .RadialBasisFunction, degree: 0, gamma: 0.5, coef0: 0.0))
        svm.train(data)
        
        //  Create a test dataset
        let testData = DataSet(dataType: .Classification, inputDimension: 2, outputDimension: 1)
        do {
            try testData.addTestDataPoint(input: [0.7, 0.6])    //  Expect 0
            try testData.addTestDataPoint(input: [0.5, 0.7])    //  Expect 0
            try testData.addTestDataPoint(input: [0.1, 0.6])    //  Expect 1
            try testData.addTestDataPoint(input: [0.1, 0.4])    //  Expect 1
            try testData.addTestDataPoint(input: [0.3, 0.1])    //  Expect 2
            try testData.addTestDataPoint(input: [0.7, 0.1])    //  Expect 2
        }
        catch {
            print("Invalid data set created")
        }
        
        //  Predict on the test data
        svm.predictValues(testData)
        
        //  See if we matched
        var classLabel : Int
        do {
            try classLabel = testData.getClass(0)
            XCTAssert(classLabel == 0, "first test data point, expect 0")
            try classLabel = testData.getClass(1)
            XCTAssert(classLabel == 0, "second test data point, expect 0")
            try classLabel = testData.getClass(2)
            XCTAssert(classLabel == 1, "third test data point, expect 1")
            try classLabel = testData.getClass(3)
            XCTAssert(classLabel == 1, "fourth test data point, expect 1")
            try classLabel = testData.getClass(4)
            XCTAssert(classLabel == 2, "fifth test data point, expect 2")
            try classLabel = testData.getClass(5)
            XCTAssert(classLabel == 2, "sixth test data point, expect 2")
        }
        catch {
            print("Error in prediction")
        }
    }
    
    func testRegression() {
        //  Create a data set - function is x1*2 - x2
        let data = DataSet(dataType: .Regression, inputDimension: 2, outputDimension: 1)
        do {
            try data.addDataPoint(input: [0.0, 1.0], output:[-1.0])
            try data.addDataPoint(input: [0.0, 0.5], output:[-0.5])
            try data.addDataPoint(input: [0.0, 0.0], output:[0.0])
            try data.addDataPoint(input: [0.5, 1.0], output:[0.0])
            try data.addDataPoint(input: [0.5, 0.5], output:[0.5])
            try data.addDataPoint(input: [0.5, 0.0], output:[1.0])
            try data.addDataPoint(input: [1.0, 1.0], output:[1.0])
            try data.addDataPoint(input: [1.0, 0.5], output:[1.5])
            try data.addDataPoint(input: [1.0, 0.0], output:[2.0])
        }
        catch {
            print("Invalid data set created")
        }
        
        //  Create an SVM regularizer and train
        let svm = SVMModel(problemType: .ϵSVMRegression, kernelSettings:
            KernelParameters(type: .RadialBasisFunction, degree: 0, gamma: 0.5, coef0: 0.0))
        svm.train(data)
        
        //  Create a test dataset - same function
        let testData = DataSet(dataType: .Regression, inputDimension: 2, outputDimension: 1)
        do {
            try testData.addTestDataPoint(input: [0.5, 0.5])    //  Expect 0.5
//            try testData.addTestDataPoint(input: [0.8, 0.0])    //  Expect 0.64
//            try testData.addTestDataPoint(input: [1.0, 0.8])    //  Expect 0.2
//            try testData.addTestDataPoint(input: [0.8, 1.0])    //  Expect -0.36
//            try testData.addTestDataPoint(input: [0.2, 0.0])    //  Expect 0.04
//            try testData.addTestDataPoint(input: [0.5, 0.1])    //  Expect 0.15
        }
        catch {
            print("Invalid data set created")
        }
        
        //  Predict on the test data
        svm.predictValues(testData)
        
        //  See if we matched
        if let testOut = testData.singleOutput(0) {
            let diff = testOut - 0.5
            XCTAssert(fabs(diff) < 0.01, "first test data point, expect -0.8")
        }
        else {
            XCTAssert(false, "first test data point, no value")
        }
    }

    func testPerformanceExample() {
        // This is an example of a performance test case.
        self.measureBlock {
            // Put the code you want to measure the time of here.
        }
    }
    func testIrisExample() {
        var irises = [Iris]()
        
//        if let path = NSBundle.mainBundle().pathForResource("iris", ofType: "json") {
            let path = "/Users/kevincoble/Desktop/swift-libsvm-feature-iris_dataset/IrisDatasetExample/iris.json"
            if let jsonData = NSData(contentsOfFile: path) {
                var jsonResult: NSArray?
                do {
                    jsonResult = try NSJSONSerialization.JSONObjectWithData(jsonData, options: .MutableContainers)  as? NSArray
                } catch  {
                    print(error)
                }
                if let jsonResult = jsonResult as? [[String : AnyObject]] {
                    for element in jsonResult {
                        irises.append(Iris(dictionary: element))
                    }
                    
                }
            }
//        }
        
        let maxPetalLength = irises.map{ $0.petalLength }.reduce(-Double.infinity, combine: { max($0, $1) })
        let minPetalLength = irises.map{ $0.petalLength }.reduce(Double.infinity, combine: { min($0, $1) })
        
        let maxPetalWidth = irises.map{ $0.petalWidth }.reduce(-Double.infinity, combine: { max($0, $1) })
        let minPetalWidth = irises.map{ $0.petalWidth }.reduce(Double.infinity, combine: { min($0, $1) })
        
        let maxSepalLength = irises.map{ $0.sepalLength }.reduce(-Double.infinity, combine: { max($0, $1) })
        let minSepalLength = irises.map{ $0.sepalLength }.reduce(Double.infinity, combine: { min($0, $1) })
        
        let maxSepalWidth = irises.map{ $0.sepalWidth }.reduce(-Double.infinity, combine: { max($0, $1) })
        let minSepalWidth = irises.map{ $0.sepalWidth }.reduce(Double.infinity, combine: { min($0, $1) })
        
        let trainData = irises
//ktc        let shuffledIrises = irises.shuffle()
//ktc        let trainCount = irises.count * 3 / 4
//ktc        let trainData = shuffledIrises[0..<trainCount]
        
//ktc        let testData = shuffledIrises[trainCount ..< irises.count]
        let testData = irises
        
        let data = DataSet(dataType: .Classification, inputDimension: 4, outputDimension: 1)
        do {
            for iris in trainData {
//ktc                let input = [
//ktc                    (iris.petalLength - minPetalLength) / (maxPetalLength - minPetalLength),
//ktc                    (iris.petalWidth - minPetalWidth) / (maxPetalWidth - minPetalWidth),
//ktc                    (iris.sepalLength - minSepalLength) / (maxSepalLength - minSepalLength),
//ktc                    (iris.sepalWidth - minSepalWidth) / (maxSepalWidth - minSepalWidth),
//ktc                ]
                let input = [iris.petalLength, iris.petalWidth, iris.sepalLength, iris.sepalWidth]
                try data.addDataPoint(input: input, output: iris.species)
            }
        }
        catch {
            print("Invalid data set created")
        }
        
        //  Create an SVM classifier and train
        let svm = SVMModel(problemType: .C_SVM_Classification, kernelSettings:
            KernelParameters(type: .RadialBasisFunction, degree: 0, gamma: 0.5, coef0: 0.0))
        svm.train(data)
        
        //  Create a test dataset
        let testDataset = DataSet(dataType: .Classification, inputDimension: 4, outputDimension: 1)
        do {
            for iris in testData {
                //ktc                let input = [
                //ktc                    (iris.petalLength - minPetalLength) / (maxPetalLength - minPetalLength),
                //ktc                    (iris.petalWidth - minPetalWidth) / (maxPetalWidth - minPetalWidth),
                //ktc                    (iris.sepalLength - minSepalLength) / (maxSepalLength - minSepalLength),
                //ktc                    (iris.sepalWidth - minSepalWidth) / (maxSepalWidth - minSepalWidth),
                //ktc                ]
                let input = [iris.petalLength, iris.petalWidth, iris.sepalLength, iris.sepalWidth]
                try testDataset.addTestDataPoint(input: input)
            }
        }
        catch {
            print("Invalid data set created")
        }
        
        //  Predict on the test data
        svm.predictValues(testDataset)
        var countCorrect = 0
        for (index, iris) in testData.enumerate() {
            let species = try! testDataset.getClass(index)
            if iris.species == species {countCorrect++}
            else { print("failed on index \(index), predicting a \(species)") }
        }
        print("percentage correct = \(Double(countCorrect) / Double(testDataset.size))")
    }

}
