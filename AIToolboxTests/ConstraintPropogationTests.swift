//
//  ConstraintPropogationTests.swift
//  AIToolbox
//
//  Created by Kevin Coble on 3/6/15.
//  Copyright (c) 2015 Kevin Coble. All rights reserved.
//

import Foundation
import XCTest
import AIToolbox

//  State data
let stateConnect : [[Int]] = [
    [1, 3],                             //  0 - WA
    [0, 3, 4, 2],                       //  1 - OR
    [1, 4, 8],                          //  2 - CA
    [5, 6, 7, 4, 1, 0],                 //  3 - ID
    [3, 7, 8, 2, 1],                    //  4 - NV
    [11, 12, 6, 3],                     //  5 - MT
    [5, 12, 13, 9, 7, 3],               //  6 - WY
    [3, 6, 9, 8, 4],                    //  7 - UT
    [7, 10, 2, 4,],                     //  8 - AZ
    [6, 13, 14, 15, 10, 7],             //  9 - CO
    [9, 15, 16, 8],                     // 10 - NM
    [17, 12, 5],                        // 11 - ND
    [11, 17, 18, 13, 6, 5],             // 12 - SD
    [12, 18, 19, 14, 9, 6],             // 13 - NE
    [13, 19, 15, 9],                    // 14 - KS
    [14, 19, 20, 16, 10, 9],            // 15 - OK
    [10, 15, 20, 21],                   // 16 - TX
    [22, 18, 12, 11],                   // 17 - MN
    [17, 22, 23, 19, 13, 12],           // 18 - IA
    [18, 23, 26, 27, 20, 15, 14, 13],   // 19 - MO
    [19, 27, 28, 21, 16, 15],           // 20 - AR
    [16, 20, 28],                       // 21 - LA
    [17, 18, 23, 24],                   // 22 - WI
    [22, 25, 26, 19, 18],               // 23 - IL
    [22, 25, 29],                       // 24 - MI
    [24, 29, 26, 23],                   // 25 - IN
    [23, 25, 29, 36, 35, 27, 19],       // 26 - KY
    [26, 35, 34, 32, 30, 28, 20, 19],   // 27 - TN
    [21, 20, 27, 30],                   // 28 - MS
    [24, 37, 36, 26, 25],               // 29 - OH
    [28, 27, 32, 31],                   // 30 - AL
    [30, 32],                           // 31 - FL
    [27, 34, 33, 31, 30],               // 32 - GA
    [34, 32],                           // 33 - SC
    [35, 33, 32, 27],                   // 34 - NC
    [34, 27, 26, 36, 47],               // 35 - VA
    [29, 37, 47, 35, 26],               // 36 - WV
    [38, 45, 46, 47, 36, 29],           // 37 - PA
    [40, 42, 44, 45, 37],               // 38 - NY
    [41],                               // 39 - ME
    [41, 42, 38],                       // 40 - VT
    [39, 42, 40],                       // 41 - NH
    [40, 41, 43, 44, 38],               // 42 - MA
    [42, 44],                           // 43 - RI
    [42, 43, 38],                       // 44 - CT
    [38, 37, 46],                       // 45 - NJ
    [45, 37, 47],                       // 46 - DE
    [37, 46, 35, 36],                   // 47 - MD
]

class ConstraintPropogationTests: XCTestCase {

    var problem = ConstraintProblem()
    var nodeList : [ConstraintProblemNode] = []
    
    override func setUp() {
        super.setUp()
        
        //  Clear lists
        problem = ConstraintProblem()
        nodeList = []
        
        //  Create each of the nodes and add them to the problem
        var nIndex = 0
        for _ in stateConnect {
            let node = ConstraintProblemNode(variableDomainSize: 4)     //  4 color map problem
            nodeList.append(node)
            nIndex += 1
        }
        problem.setNodeList(nodeList)
        
        //  Create each of the constraints and add them to the nodes
        for stateIndex in 0..<stateConnect.count {
            for touchingState in stateConnect[stateIndex] {
                problem.addConstraintOfType(.cantBeSameValueInOtherNode, betweenNodeIndex: stateIndex, andNodeIndex: touchingState)
            }
        }
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testForward() {
        let result = problem.solveWithForwardPropogation()
        XCTAssert(result, "Forward propogation did not find solution")
        
        for node in nodeList {
            XCTAssert(node.variableIndexValue != nil, "Forward propogation solution has unset node")
        }
        
        for stateIndex in 0..<stateConnect.count {
            for touchingState in stateConnect[stateIndex] {
                let different = (nodeList[stateIndex].variableIndexValue != nodeList[touchingState].variableIndexValue)
                XCTAssert(different, "Forward propogation solution has touching states with same variable index")
            }
        }
    }
    
    func testSingleton() {
        let result = problem.solveWithSingletonPropogation()
        XCTAssert(result, "Singleton propogation did not find solution")
        
        for node in nodeList {
            XCTAssert(node.variableIndexValue != nil, "Singleton propogation solution has unset node")
        }
        
        for stateIndex in 0..<stateConnect.count {
            for touchingState in stateConnect[stateIndex] {
                let different = (nodeList[stateIndex].variableIndexValue != nodeList[touchingState].variableIndexValue)
                XCTAssert(different, "Singleton propogation solution has touching states with same variable index")
            }
        }
    }
    
    func testFull() {
        let result = problem.solveWithFullPropogation()
        XCTAssert(result, "Full propogation did not find solution")
        
        for node in nodeList {
            XCTAssert(node.variableIndexValue != nil, "Full propogation solution has unset node")
        }
        
        for stateIndex in 0..<stateConnect.count {
            for touchingState in stateConnect[stateIndex] {
                let different = (nodeList[stateIndex].variableIndexValue != nodeList[touchingState].variableIndexValue)
                XCTAssert(different, "Full propogation solution has touching states with same variable index")
            }
        }
    }

    func testPerformanceExample() {
        // This is an example of a performance test case.
        self.measure() {
            // Put the code you want to measure the time of here.
        }
    }

}
