//
//  TreeTests.swift
//  AIToolbox
//
//  Created by Kevin Coble on 2/15/15.
//  Copyright (c) 2015 Kevin Coble. All rights reserved.
//

import Cocoa
import XCTest
import AIToolbox

class TestTreeNode : TreeNode {
    var nodeList : [TestTreeNode] = []
    var getChildNodeList : [TreeNode] { return nodeList}
    
    var value: Int
    
    init(startValue: Int) {
        value = startValue
    }
    
    func addChild(child: TestTreeNode) {
        nodeList.append(child)
    }
    
    func getChildNodeForIndex(index: Int) -> TreeNode? {
        if (index < nodeList.count) {
            return nodeList[index]
        }
        return nil
    }
}

class TreeTests: XCTestCase {
    
    var tree : Tree<TestTreeNode> = Tree<TestTreeNode>(initRootNode: TestTreeNode(startValue: 0))
    var A = TestTreeNode(startValue: 1)

    override func setUp() {
        super.setUp()
        
        //  Build a tree
/*
             root
            /    \
           A      B
          / \    / \
         C   D  E   F
*/
        var B = TestTreeNode(startValue: 2)
        var C = TestTreeNode(startValue: 3)
        var D = TestTreeNode(startValue: 4)
        var E = TestTreeNode(startValue: 5)
        var F = TestTreeNode(startValue: 6)
        tree.getRootNode.addChild(A)
        tree.getRootNode.addChild(B)
        A.addChild(C)
        A.addChild(D)
        B.addChild(E)
        B.addChild(F)
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testContainsNodeDepthFirst() {
        var result = tree.containsNodeDepthFirst(nil, searchCriteria: {$0.value == 4})    //  Look for the tree to contain 4
        XCTAssert(result, "Finds child node that has given value")
        result = tree.containsNodeDepthFirst(nil, searchCriteria: {$0.value == 8})    //  Look for the tree to contain 8
        XCTAssert(!result, "Does not find child node that has unused value")
        result = tree.containsNodeDepthFirst(A, searchCriteria: {$0.value == 3})    //  Look for the tree to contain 3, from branch that does go down that path
        XCTAssert(result, "Does not find child node that has unused value")
        result = tree.containsNodeDepthFirst(A, searchCriteria: {$0.value == 5})    //  Look for the tree to contain 5, from branch that doesn't go down that path
        XCTAssert(!result, "Does not find child node that has unused value")
    }

    func testPerformanceExample() {
        // This is an example of a performance test case.
        self.measureBlock() {
            // Put the code you want to measure the time of here.
            let result = tree.containsNodeDepthFirst(nil, searchCriteria: {$0.value == 6})    //  Look for the tree to contain 6
        }
    }

}
