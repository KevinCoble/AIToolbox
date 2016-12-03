//
//  AlphaBetaTests.swift
//  AIToolbox
//
//  Created by Kevin Coble on 2/23/15.
//  Copyright (c) 2015 Kevin Coble. All rights reserved.
//

import Foundation
import XCTest
import AIToolbox

/*
values at each branch            root

max                              8
.                  /------------/ \------------\
.                 /                             \
min              7                               8
.               / \                             / \
.          /---/   \---\                   /---/   \---\
.         /             *                 /             *
max      7               8               8               9
.      /   \           /   \           /   \           /   \
.     /     *         /     \         *     \         /     \
min  7       3       8       0       1       8       9       0
.   / \     / *     / \     / \     / *     / \     / \     / \
.  8   7   3   0   9   8   0   0   1   0   8   9   9   9   0   0
0 indicates a node not used

order of creation

max                              0
.                  /------------/ \------------\
.                 /                             \
min              1                               2
.               / \                             / \
.          /---/   \---\                   /---/   \---\
.         /             *                 /             *
max      3               4              15              16
.      /   \           /   \           /   \           /   \
.     /     *         /     \         *     \         /     \
min  5       6      11      12       17     18       23      24
.   / \     / *     / \     / \     / *     / \     / \     / \
.  7   8   9  10  13  14   0   0   19  20  21  22 25   26  0   0

*/
var staticEvaluationIndex = 0
//                            0  1  2  3  4  5  6  7  8  9  0  1  2  3  4  5  6  7  8  9  0  1  2  3  4  5  6  7  8  9
let staticEvaluationValues = [0, 0, 0, 0, 0, 0, 0, 8, 7, 3, 0, 0, 0, 9, 8, 0, 0, 0, 0, 1, 0, 8, 9, 0, 0, 9, 9, 0, 0, 0]

class TestAlphaBetaNode : AlphaBetaNode {
    
    let creationIndex : Int
    init() {
        creationIndex = staticEvaluationIndex
        staticEvaluationIndex += 1
        print("created node \(creationIndex)")
    }
    
    func generateMoves(_ forMaximizer: Bool) -> [AlphaBetaNode] {
        var returnMoves : [AlphaBetaNode] = []
        print("created children of \(creationIndex)")
        returnMoves.append(TestAlphaBetaNode())
        returnMoves.append(TestAlphaBetaNode())
        return returnMoves
    }
    func staticEvaluation() -> Double {
        let value = Double(staticEvaluationValues[creationIndex])
        print("static evaluation of \(creationIndex) resulting in \(value)")
        return value
    }
}


class AlphaBetaTests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }

    func testExample() {
        // This is an example of a functional test case.
        let graph = AlphaBetaGraph()
        staticEvaluationIndex = 0
        let startNode = TestAlphaBetaNode()
        let result = graph.startAlphaBetaWithNode(startNode, forDepth: 4, startingAsMaximizer: true) as! TestAlphaBetaNode?
        XCTAssert(result?.creationIndex == 2, "alpha-beta")
    }

    func testPerformanceExample() {
        // This is an example of a performance test case.
        self.measure() {
            // Put the code you want to measure the time of here.
        }
    }

}
