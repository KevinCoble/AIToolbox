//
//  GraphTests.swift
//  AIToolbox
//
//  Created by Kevin Coble on 2/15/15.
//  Copyright (c) 2015 Kevin Coble. All rights reserved.
//

import Foundation
import XCTest
import AIToolbox

class TestGraphNode : GraphNode {
    var value : Int
    init (initValue : Int) {
        value = initValue
    }
}


class GraphTests: XCTestCase {
    
    var graph : Graph<TestGraphNode> = Graph<TestGraphNode>()
    var nodes = [TestGraphNode]()

    override func setUp() {
        super.setUp()
        
        //  Create a graph nodes.  Value is approximate distance to node F (vertical/horizontal hop = 10, diagonal = 14)
        let distance = [32, 22, 14, 20, 10, 0, 10, 14, 20, 22]
        for i in 0..<10 {
            nodes.append(TestGraphNode(initValue: distance[i]))
            graph.addNode(nodes[i])
        }
        
/*
        .                           [I]--3->[J]
        .                            ^
        .                            |
        .                            8
        .                            |
        .                            |
        .   [A]--1->[B]--2->[C]<-2--[G]--2->[H]
        .     \              |     ^
        .      \             |    /
        .       4            2   3
        .        \           |  /
        .         v          v /
        .           [D]--2->[E]--1->[F]
*/
        nodes[0].addEdge(GraphEdge(destination: nodes[1], costValue: 1.0))
        nodes[1].addEdge(GraphEdge(destination: nodes[2], costValue: 2.0))
        nodes[0].addEdge(GraphEdge(destination: nodes[3], costValue: 4.0))
        nodes[3].addEdge(GraphEdge(destination: nodes[4], costValue: 2.0))
        nodes[2].addEdge(GraphEdge(destination: nodes[4], costValue: 2.0))
        nodes[4].addEdge(GraphEdge(destination: nodes[5], costValue: 1.0))
        nodes[4].addEdge(GraphEdge(destination: nodes[6], costValue: 3.0))
        nodes[6].addEdge(GraphEdge(destination: nodes[2], costValue: 2.0))
        nodes[6].addEdge(GraphEdge(destination: nodes[7], costValue: 2.0))
        nodes[6].addEdge(GraphEdge(destination: nodes[8], costValue: 8.0))
        nodes[8].addEdge(GraphEdge(destination: nodes[9], costValue: 3.0))
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }

    func testDepthFirstSearch() {
        var result: [Int]?
        
        //  Test case where there is no path
        result = graph.getDepthFirstPath(fromNode: nodes[1], searchCriteria: {$0 === self.nodes[0]})
        XCTAssert(result == nil, "Depth First search, No path")
    
        //  Test case where start node is goal node
        result = graph.getDepthFirstPath(fromNode: nodes[0], searchCriteria: {$0 === self.nodes[0]})
        if let result = result {
            XCTAssert(result.count == 0, "Depth First search, start node is goal node")
        }
        else {
            XCTAssert(false, "Depth First search, start node is goal node")
        }
        
        //  Test case where there is a path
        result = graph.getDepthFirstPath(fromNode: nodes[0], searchCriteria: {$0 === self.nodes[4]})
        if let result = result {
            XCTAssert(result.count == 3, "Depth First search, start node to E node")
            XCTAssert(result[0] == 0 && result[1] == 0 && result[2] == 0, "Depth First search, start node to E node")
            
            //  Get the second path
            let secondResult = graph.getDepthFirstNextPath(searchCriteria: {$0 === self.nodes[4]})
//!!            let secondResult = graph.getDepthFirstNextPath(fromNode: nodes[0], lastPath: result, searchCriteria: {$0 === self.nodes[4]})
            if let secondResult = secondResult {
                XCTAssert(secondResult.count == 2, "Depth First search, second path from start node to E node")
                XCTAssert(secondResult[0] == 1 && secondResult[1] == 0, "Depth First search, second path from start node to E node")
                
                //  Verify no third path
//!!                let thirdResult = graph.getDepthFirstNextPath(fromNode: nodes[0], lastPath: result, searchCriteria: {$0 === self.nodes[4]})
                let thirdResult = graph.getDepthFirstNextPath(searchCriteria: {$0 === self.nodes[4]})
                XCTAssert(thirdResult == nil, "Depth First search, third path start node to E node")
            }
            
        }
        else {
            XCTAssert(false, "Depth First search, start node to last node")
        }
    }

    func testDepthFirstPerformance() {
        // This is an example of a performance test case.
        self.measure() {
            // Put the code you want to measure the time of here.
            _ = self.graph.getDepthFirstPath(fromNode: self.nodes[0], searchCriteria: {$0 === self.nodes[9]})
//!!            self.graph.getDepthFirstNextPath(fromNode: self.nodes[0], lastPath: result, searchCriteria: {$0 === self.nodes[9]})
            _ = self.graph.getDepthFirstNextPath(searchCriteria: {$0 === self.nodes[9]})
        }
    }

    func testBreadthFirstSearch() {
        var result: [Int]?
        
        //  Test case where there is no path
        result = graph.getBreadthFirstPath(fromNode: nodes[1], searchCriteria: {$0 === self.nodes[0]})
        XCTAssert(result == nil, "Breadth First search, No path")
        
        //  Test case where start node is goal node
        result = graph.getBreadthFirstPath(fromNode: nodes[0], searchCriteria: {$0 === self.nodes[0]})
        if let result = result {
            XCTAssert(result.count == 0, "Breadth First search, start node is goal node")
        }
        else {
            XCTAssert(false, "Breadth First search, start node is goal node")
        }
        
        //  Test case where there is a path
        result = graph.getBreadthFirstPath(fromNode: nodes[0], searchCriteria: {$0 === self.nodes[4]})
        if let result = result {
            XCTAssert(result.count == 2, "Breadth First search, start node to E node")
            XCTAssert(result[0] == 1 && result[1] == 0, "Breadth First search, start node to E node")
            
            //  Get the second path
            let secondResult = graph.getBreadthFirstNextPath(searchCriteria: {$0 === self.nodes[4]})
            if let secondResult = secondResult {
                XCTAssert(secondResult.count == 3, "Breadth First search, second path from start node to E node")
                XCTAssert(secondResult[0] == 0 && secondResult[1] == 0 && secondResult[2] == 0, "Breadth First search, second path from start node to E node")
                
                //  Verify no third path
                let thirdResult = graph.getBreadthFirstNextPath(searchCriteria: {$0 === self.nodes[4]})
                XCTAssert(thirdResult == nil, "Breadth First search, third path start node to E node")
            }
        }
        else {
            XCTAssert(false, "Breadth First search, start node to last node")
        }
    }
    
    func testBreadthFirstPerformance() {
        // This is an example of a performance test case.
        self.measure() {
            // Put the code you want to measure the time of here.
            _ = self.graph.getBreadthFirstPath(fromNode: self.nodes[0], searchCriteria: {$0 === self.nodes[9]})
            _ = self.graph.getBreadthFirstNextPath(searchCriteria: {$0 === self.nodes[9]})
        }
    }
    
    func testHillClimbSearch() {
        var result: [Int]?
        
        //  Test case where there is no path
        result = graph.getHillClimbPath(fromNode: nodes[1], nodeHeuristic: {-Double($0.value)}, searchCriteria: {$0 === self.nodes[0]})
        XCTAssert(result == nil, "Hill Climb search, No path")
        
        //  Test case where start node is goal node
        result = graph.getHillClimbPath(fromNode: nodes[0], nodeHeuristic: {-Double($0.value)}, searchCriteria: {$0 === self.nodes[0]})
        if let result = result {
            XCTAssert(result.count == 0, "Hill Climb search, start node is goal node")
        }
        else {
            XCTAssert(false, "Hill Climb search, start node is goal node")
        }
        
        //  Test case where there is a path
        result = graph.getHillClimbPath(fromNode: nodes[0], nodeHeuristic: {-Double($0.value)}, searchCriteria: {$0 === self.nodes[5]})
        if let result = result {
            XCTAssert(result.count == 3, "Hill Climb search, start node to F node")
            XCTAssert(result[0] == 1 && result[1] == 0 && result[2] == 0, "Hill Climb search, start node to F node")
            
            //  Get the second path
            let secondResult = graph.getHillClimbNextPath(nodeHeuristic: {-Double($0.value)}, searchCriteria: {$0 === self.nodes[5]})
            if let secondResult = secondResult {
                XCTAssert(secondResult.count == 4, "Hill Climb search, second path from start node to F node")
                XCTAssert(secondResult[0] == 0 && secondResult[1] == 0 && secondResult[2] == 0 && secondResult[2] == 0, "Hill Climb search, second path from start node to E node")
                
                //  Verify no third path
                let thirdResult = graph.getHillClimbNextPath(nodeHeuristic: {-Double($0.value)}, searchCriteria: {$0 === self.nodes[5]})
                XCTAssert(thirdResult == nil, "Hill Climb search, third path start node to F node")
            }
            
        }
        else {
            XCTAssert(false, "Hill Climb search, start node to last node")
        }
    }
    
    func testHillClimbPerformance() {
        // This is an example of a performance test case.
        self.measure() {
            // Put the code you want to measure the time of here.
            _ = self.graph.getHillClimbPath(fromNode: self.nodes[0], nodeHeuristic: {-Double($0.value)}, searchCriteria: {$0 === self.nodes[9]})
            _ = self.graph.getHillClimbNextPath(nodeHeuristic: {-Double($0.value)}, searchCriteria: {$0 === self.nodes[9]})
        }
    }
    
    func testBeamSearch() {
        var result: [Int]?
        
        //  Test case where there is no path
        result = graph.getBeamPath(fromNode: nodes[1], nodeHeuristic: {-Double($0.value)}, beamWidth : 2, searchCriteria: {$0 === self.nodes[0]})
        XCTAssert(result == nil, "Beam search, No path")
        
        //  Test case where start node is goal node
        result = graph.getBeamPath(fromNode: nodes[0], nodeHeuristic: {-Double($0.value)}, beamWidth : 2, searchCriteria: {$0 === self.nodes[0]})
        if let result = result {
            XCTAssert(result.count == 0, "Beam search, start node is goal node")
        }
        else {
            XCTAssert(false, "Beam search, start node is goal node")
        }
        
        //  Test case where there is a path
        result = graph.getBeamPath(fromNode: nodes[0], nodeHeuristic: {-Double($0.value)}, beamWidth : 2, searchCriteria: {$0 === self.nodes[4]})
        if let result = result {
            XCTAssert(result.count == 2, "Beam search, start node to E node")
            XCTAssert(result[0] == 1 && result[1] == 0, "Beam search, start node to E node")
            
        }
        else {
            XCTAssert(false, "Beam search, start node to last node")
        }
    }
    
    func testOptimalSearch() {
        var result: [Int]?
        
        //  Test case where there is no path
        result = graph.getShortestPathByBestFirst(fromNode: nodes[1], searchCriteria: {$0 === self.nodes[0]})
        XCTAssert(result == nil, "Best-First search, No path")
        
        //  Test case where start node is goal node
        result = graph.getShortestPathByBestFirst(fromNode: nodes[0], searchCriteria: {$0 === self.nodes[0]})
        if let result = result {
            XCTAssert(result.count == 0, "Best-First search, start node is goal node")
        }
        else {
            XCTAssert(false, "Best-First search, start node is goal node")
        }
        
        //  Test case where there is a path
        result = graph.getShortestPathByBestFirst(fromNode: nodes[0], searchCriteria: {$0 === self.nodes[4]})
        if let result = result {
            XCTAssert(result.count == 3, "Best-First search, start node to E node")
            XCTAssert(result[0] == 0 && result[1] == 0 && result[2] == 0, "Best-First search, start node to E node")
            
        }
        else {
            XCTAssert(false, "Best-First search, start node to last node")
        }
        
        //  Test case where there is no path
        result = graph.getShortestPathByAStar(fromNode: nodes[1], destinationNode: nodes[0])
        XCTAssert(result == nil, "A* search, No path")
        
        //  Test case where start node is goal node
        result = graph.getShortestPathByAStar(fromNode: nodes[0], destinationNode: nodes[0])
        if let result = result {
            XCTAssert(result.count == 0, "A* search, start node is goal node")
        }
        else {
            XCTAssert(false, "A* search, start node is goal node")
        }
        
        //  Test case where there is a path
        result = graph.getShortestPathByAStar(fromNode: nodes[0], destinationNode: nodes[4])
        if let result = result {
            XCTAssert(result.count == 3, "A* search, start node to E node")
            XCTAssert(result[0] == 0 && result[1] == 0 && result[2] == 0, "A* search, start node to E node")
            
        }
        else {
            XCTAssert(false, "A* search, start node to last node")
        }
    }
}
