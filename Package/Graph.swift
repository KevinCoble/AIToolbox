//
//  Graph.swift
//  AIToolbox
//
//  Created by Kevin Coble on 2/15/15.
//  Copyright (c) 2015 Kevin Coble. All rights reserved.
//

import Foundation

// MARK: GraphEdge
///  Class for link between GraphNodes.
///  Only subclass if you need to override the cost function to be variable
open class GraphEdge {
    let destinationNode : GraphNode
    let cost : Double
    
    required public init(destination : GraphNode) {
        //  Save the destination
        destinationNode = destination
        
        //  Set the cost to 0
        cost = 0.0
    }
    
    public init (destination: GraphNode, costValue: Double) {
        //  Save the destination
        destinationNode = destination
        
        //  Save the cost
        cost = costValue
    }
    
    ///  Method to get cost
    ///  Override to make calculation.  Edge is from passed in sourceNode to stored destinationNode
    open func getCost(_ sourceNode: GraphNode) -> Double {return cost}
    
    ///  Method to determine if this edge has the passed in GraphNode as a destination
    open func goesToNode(_ node: GraphNode) -> Bool {
        return node === destinationNode
    }
}

// MARK: -
// MARK: GraphNode
///  Subclass GraphNode to add data to a graph node
open class GraphNode {
    var edgeList: [GraphEdge] = []
    var visited = false
    var leastCostToNode = Double.infinity
    
    //  Initializer
    public init() {
        
    }
    
    ///  Method to add a GraphEdge (or subclass) to a node
    open func addEdge(_ newEdge: GraphEdge) {
        edgeList.append(newEdge)
    }
    
    ///  Convenience method to add two edges, making a bi-directional connection.
    ///  This only works with standard GraphEdge classes, not sub-classes.
    ///  If the optional second cost is not provided, the passed-in edge cost is duplicated
    open func addBidirectionalEdge(_ newEdge: GraphEdge, reverseCost: Double?) {
        //  Add the first direction
        edgeList.append(newEdge)
        
        //  Create the edge for the other direction
        var edge : GraphEdge
        if let cost = reverseCost {
            edge = GraphEdge(destination: self, costValue: cost)
        }
        else {
            edge = GraphEdge(destination: self, costValue: newEdge.cost)
        }
        
        //  Add the edge to the other node
        newEdge.destinationNode.addEdge(edge)
    }
    
    ///  Method to remove all edges to a specified destination node
    open func removeEdgesToNode(_ node: GraphNode) {
        var index = edgeList.count-1
        while (index >= 0) {
            if edgeList[index].goesToNode(node) {
                edgeList.remove(at: index)
            }
            index -= 1
        }
    }
    
    ///  Method to return the admissible heuristic value for an A* search
    ///  The value is a lower-bound estimate of the remaining cost to the goal
    open func admissibleHeuristicCost() -> Double { return 0.0 }
}

// MARK: -
// MARK: Search Queue Entry
//  Struct for queueing breadth-first nodes and search information on the queue
struct SearchQueueEntry<T> {
    let graphNode : T
    var pathToNode : [Int] = []
    var costToNode : Double
    
    init(node: T) {
        graphNode = node
        costToNode = 0.0
    }
    
    init(node: T, cost: Double) {
        graphNode = node
        costToNode = cost
    }
}


// MARK: -
// MARK: Graph Class
///  Class for a graph of nodes.  The nodes are subclasses of GraphNode, connected by
///    GraphEdge classes (or subclasses)
open class Graph<T : GraphNode> {
    //  Array of nodes in the list
    var graphNodeList : [T] = []
    
    //  Current path in depth-first search
    var currentPath: [Int] = []
    
    //  'Queue' for breadth-first search.  These are two arrays to allow sorting for pruning metheds
    var currentQueueList = [SearchQueueEntry<T>]()
    var nextDepthList = [SearchQueueEntry<T>]()
    
    //  Empty initializer
    public init() {
        
    }
    
    ///  Method to add a GraphNode (or subclass) to a graph
    open func addNode(_ newNode: T) {
        graphNodeList.append(newNode)
    }
    
    ///  Method to remove a node from the graph, along with all edges to that node
    open func removeNode(_ nodeToRemove: T) {
        for index in 0 ..< graphNodeList.count  {
            if graphNodeList[index] === nodeToRemove {
                graphNodeList.remove(at: index)
                for node in graphNodeList {
                    node.removeEdgesToNode(nodeToRemove)
                }
                return
            }
        }
    }
    
    
    // MARK: Depth-First
    ///  Method to do depth-first search of graph to find path between nodes.
    ///  Returns an array of edge indices for the traverse path from the start to the end node.
    ///  Returns empty array if start and end nodes are the same.  Returns nil if no path exists
    open func getDepthFirstPath(fromNode: T, searchCriteria: (T) -> Bool) -> [Int]? {
        //  Clear the 'queue' arrays
        currentQueueList = []
        nextDepthList = []
        
        //  Clear the visited flags
        for node in graphNodeList { node.visited = false }
        
        //  Push the first node
        currentQueueList.append(SearchQueueEntry<T>(node:fromNode))
        
        //  Mark the first node as visited
        fromNode.visited = true
        
        //  Use the 'find next path' function to find the path
        return getDepthFirstNextPath(searchCriteria: searchCriteria)
    }
    
    ///  Method to do depth-first search of graph to find path between nodes, starting at the last path found.
    ///  Returns an array of edge indices for the traverse path from the start to the end node.
    ///  Returns empty array if start and end nodes are the same.  Returns nil if no path exists.
    ///  Modifying the graph between calls will result in unexpected behavior - possibly loops
    open func getDepthFirstNextPath(searchCriteria: (T) -> Bool) -> [Int]? {
        //  Iterate until we find a path, or we run out of queue
        while (true) {
            if currentQueueList.count > 0 {
                //  Get the next entry for the current depth
                let entry = currentQueueList.removeLast()
                
                //  Push each of the children onto the next level
                var index = 0
                for edge in entry.graphNode.edgeList {
                    if (!edge.destinationNode.visited) {
                        var newEntry = SearchQueueEntry<T>(node: (edge.destinationNode as! T))
                        newEntry.pathToNode = entry.pathToNode
                        newEntry.pathToNode.append(index)
                        nextDepthList.append(newEntry)
                        edge.destinationNode.visited = true
                    }
                    index += 1
                }
                
                //  Put the next level at the end of the current queue
                currentQueueList += Array(nextDepthList.reversed())  //  reverse since we are taking last item off)
                nextDepthList = []
                
                //  See if this is the node we want
                if (searchCriteria(entry.graphNode)) {return  entry.pathToNode}
            }
            else {
                return nil
            }
        }
    }
    
    // MARK: Breadth-First
    ///  Method to do breadth-first search of graph to find path between nodes.
    ///  Returns an array of edge indices for the traverse path from the start to the end node.
    ///  Returns empty array if start and end nodes are the same.  Returns nil if no path exists
    open func getBreadthFirstPath(fromNode: T, searchCriteria: (T) -> Bool) -> [Int]? {
        //  Clear the 'queue' arrays
        currentQueueList = []
        nextDepthList = []
        
        //  Clear the visited flags
        for node in graphNodeList { node.visited = false }
        
        //  Push the first node
        currentQueueList.append(SearchQueueEntry<T>(node:fromNode))
        
        //  Mark the first node as visited
        fromNode.visited = true
        
        //  Use the 'find next path' function to find the path
        return getBreadthFirstNextPath(searchCriteria: searchCriteria)
    }
    
    ///  Method to do breadth-first search of graph to find path between nodes, starting at the last path found.
    ///  Call getBreadthFirstPath to start search at beginning
    ///  Returns an array of edge indices for the traverse path from the start to the end node.
    ///  Returns empty array if start and end nodes are the same.  Returns nil if no path exists.
    ///  Modifying the graph between calls will result in unexpected behavior - possibly loops
    open func getBreadthFirstNextPath(searchCriteria: (T) -> Bool) -> [Int]? {
        //  Iterate until we find a path, or we run out of queue
        while (true) {
            if currentQueueList.count > 0 {
                //  Get the next entry for the current depth
                let entry = currentQueueList.removeLast()
                
                //  Push each of the children onto the next level
                var index = 0
                for edge in entry.graphNode.edgeList {
                    if (!edge.destinationNode.visited) {
                        var newEntry = SearchQueueEntry<T>(node: (edge.destinationNode as! T))
                        newEntry.pathToNode = entry.pathToNode
                        newEntry.pathToNode.append(index)
                        nextDepthList.append(newEntry)
                        edge.destinationNode.visited = true
                    }
                    index += 1
                }
                
                //  See if this is the node we want
                if (searchCriteria(entry.graphNode)) {return  entry.pathToNode}
            }
            else {
                if (nextDepthList.count == 0) {return nil}  //  No more nodes, return nil to indicate no path
                currentQueueList = Array(nextDepthList.reversed())  //  Go down one more depth level (reverse since we are taking last item off)
                nextDepthList = []
            }
        }
    }
    
    
    // MARK: Hill Climb
    ///  Method to do hill-climb search of graph to find path between nodes.
    ///  The heuristic is a closure/function that returns the relative value of the node compared to the goal.  Larger values indicate closer to the goal.
    ///  Returns an array of edge indices for the traverse path from the start to the end node.
    ///  Returns empty array if start and end nodes are the same.  Returns nil if no path exists
    open func getHillClimbPath(fromNode: T, nodeHeuristic: (T) -> Double, searchCriteria: (T) -> Bool) -> [Int]? {
        //  Clear the 'queue' arrays
        currentQueueList = []
        nextDepthList = []
        
        //  Clear the visited flags
        for node in graphNodeList { node.visited = false }
        
        //  Push the first node
        currentQueueList.append(SearchQueueEntry<T>(node:fromNode))
        
        //  Mark the first node as visited
        fromNode.visited = true
        
        //  Use the 'find next path' function to find the path
        return getHillClimbNextPath(nodeHeuristic: nodeHeuristic, searchCriteria: searchCriteria)
    }
    
    ///  Method to do hill-climb search of graph to find path between nodes, starting at the last path found.
    ///  Returns an array of edge indices for the traverse path from the start to the end node.
    ///  Returns empty array if start and end nodes are the same.  Returns nil if no path exists.
    ///  Modifying the graph between calls will result in unexpected behavior - possibly loops
    open func getHillClimbNextPath(nodeHeuristic: (T) -> Double, searchCriteria: (T) -> Bool) -> [Int]? {
        //  Iterate until we find a path, or we run out of queue
        while (true) {
            if currentQueueList.count > 0 {
                //  Get the next entry for the current depth
                let entry = currentQueueList.removeLast()
                
                //  Push each of the children onto the next level
                var index = 0
                for edge in entry.graphNode.edgeList {
                    if (!edge.destinationNode.visited) {
                        var newEntry = SearchQueueEntry<T>(node: (edge.destinationNode as! T))
                        newEntry.pathToNode = entry.pathToNode
                        newEntry.pathToNode.append(index)
                        edge.destinationNode.visited = true
                        
                        //  Add the new node sorted, based on it's heuristic value
                        _ = enqueueCurrentOrdered(newEntry, orderHeuristic: nodeHeuristic)
                    }
                    index += 1
                }
                
                //  See if this is the node we want
                if (searchCriteria(entry.graphNode)) {return  entry.pathToNode}
            }
            else {
                return nil
            }
        }
    }
    
    //  Enqueue on the current list with the higher heuristic going to the end
    func enqueueCurrentOrdered(_ newItem: SearchQueueEntry<T>, orderHeuristic: (T) -> Double) -> Bool {
        var max = currentQueueList.count
        if (max == 0) {
            currentQueueList.append(newItem)
            return true
        }
        var min = 0
        let newItemHeuristicValue = orderHeuristic(newItem.graphNode)
        
        var mid : Int
        while (min < max) {
            mid = (min + max) / 2
            if (orderHeuristic(currentQueueList[mid].graphNode) < newItemHeuristicValue) {
                min = mid + 1;
            }
            else {
                max = mid;
            }
        }
        if (max == min) {
            if (max != currentQueueList.count && orderHeuristic(currentQueueList[max].graphNode) == newItemHeuristicValue) {
                return false
            }
            currentQueueList.insert(newItem, at: max)
            return true
        }
        else {
            return false
        }
    }
    
    
    // MARK: Beam Search
    ///  Method to do beam search of graph to find path between nodes.  The width of the 'beam' is a parameter.
    ///  The heuristic is a closure/function that returns the relative value of the node compared to the goal.  Larger values indicate closer to the goal.
    ///  Returns an array of edge indices for the traverse path from the start to the end node.
    ///  Returns empty array if start and end nodes are the same.  Returns nil if no path exists
    open func getBeamPath(fromNode: T, nodeHeuristic: (T) -> Double, beamWidth: Int, searchCriteria: (T) -> Bool) -> [Int]? {
        //  Clear the 'queue' arrays
        currentQueueList = []
        nextDepthList = []
        
        //  Validate the input
        if (beamWidth < 1) {return nil}
        
        //  Clear the visited flags
        for node in graphNodeList { node.visited = false }
        
        //  Push the first node
        currentQueueList.append(SearchQueueEntry<T>(node:fromNode))
        
        //  Mark the first node as visited
        fromNode.visited = true

        //  Iterate until we find a path, or we run out of queue
        while (true) {
            if currentQueueList.count > 0 {
                //  Get the next entry for the current depth
                let entry = currentQueueList.removeLast()
                
                //  Push each of the children onto the next level
                var index = 0
                for edge in entry.graphNode.edgeList {
                    if (!edge.destinationNode.visited) {
                        var newEntry = SearchQueueEntry<T>(node: (edge.destinationNode as! T))
                        newEntry.pathToNode = entry.pathToNode
                        newEntry.pathToNode.append(index)
                        nextDepthList.append(newEntry)
                        edge.destinationNode.visited = true
                    }
                    index += 1
                }
                
                //  See if this is the node we want
                if (searchCriteria(entry.graphNode)) {return  entry.pathToNode}
            }
            else {
                if (nextDepthList.count == 0) {return nil}  //  No more nodes, return nil to indicate no path
                nextDepthList.sort(by: {nodeHeuristic($0.graphNode) > nodeHeuristic($1.graphNode)})     //  Sort best first
                while (nextDepthList.count > beamWidth) { nextDepthList.removeLast() }              //  Limit to the beam width
                currentQueueList = Array(nextDepthList.reversed())  //  Go down one more depth level (reverse since we are taking last item off)
                nextDepthList = []
            }
        }
    }
    
    
    // MARK: Optimal Paths
    ///  Method to do a best-first search for the shortest path between the start and goal.
    ///  The 'length' of each edge traversal is retrieved using the 'getCost' method.
    ///  Returns an array of edge indices for the traverse path from the start to the end node.
    ///  Returns empty array if start and end nodes are the same.  Returns nil if no path exists
    open func getShortestPathByBestFirst(fromNode: T, searchCriteria: (T) -> Bool) -> [Int]? {
        //  Clear the 'queue' arrays
        currentQueueList = []
        nextDepthList = []
        
        //  Set the cost to each node to infinite
        for node in graphNodeList { node.leastCostToNode = Double.infinity }
        
        //  Push the first node
        currentQueueList.append(SearchQueueEntry<T>(node:fromNode))
        
        //  Mark the first node as visited with no cost
        fromNode.leastCostToNode = 0.0
        
        //  Start with a best-goal cost of infinity
        var bestGoalCost = Double.infinity
        var result : [Int]?
        
        //  Iterate until we find a path, or we run out of queue
        while (true) {
            if currentQueueList.count > 0 {
                //  Get the next entry for the current depth
                let entry = currentQueueList.removeLast()
                
                //  Push each of the children onto the next level
                var index = 0
                for edge in entry.graphNode.edgeList {
                    let newCost = entry.costToNode + edge.getCost(entry.graphNode)
                    if (newCost < edge.destinationNode.leastCostToNode) {
                        if (newCost < bestGoalCost) {   //  Skip any that are already have higher cost than a found goal
                            var newEntry = SearchQueueEntry<T>(node: (edge.destinationNode as! T), cost: newCost)
                            newEntry.pathToNode = entry.pathToNode
                            newEntry.pathToNode.append(index)
                            _ = enqueueOrderedByCost(newEntry)
                            edge.destinationNode.leastCostToNode = newCost
                        }
                    }
                    index += 1
                }
                
                //  See if this is a goal node
                if (searchCriteria(entry.graphNode)) {
                    //  See if it is better than any other goal node we found at this depth
                    if (entry.costToNode < bestGoalCost) {
                        bestGoalCost = entry.costToNode
                        result = entry.pathToNode
                    }
                }
            }
            else {
                //  Done checking, return the result
                return result
            }
        }
    }
    ///  Method to do an A* search for the shortest path between the start and destination node.
    ///  The 'length' of each edge traversal is retrieved using the 'getCost' method.
    ///  The admissible heuristic is retrieved using the 'getAdmissibleHeuristic' method
    ///  Returns an array of edge indices for the traverse path from the start to the end node.
    ///  Returns empty array if start and end nodes are the same.  Returns nil if no path exists
    open func getShortestPathByAStar(fromNode: T, destinationNode: T) -> [Int]? {
        //  Clear the 'queue' arrays
        currentQueueList = []
        nextDepthList = []
        
        //  Set the cost to each node to infinite
        for node in graphNodeList { node.leastCostToNode = Double.infinity }
        
        //  Push the first node
        currentQueueList.append(SearchQueueEntry<T>(node:fromNode))
        
        //  Mark the first node as visited with no cost
        fromNode.leastCostToNode = 0.0
        
        //  Start with a best-goal cost of infinity
        var bestGoalCost = Double.infinity
        var result : [Int]?
        
        //  Iterate until we find a path, or we run out of queue
        while (true) {
            if currentQueueList.count > 0 {
                //  Get the next entry for the current depth
                let entry = currentQueueList.removeLast()
                let actualCost = entry.costToNode - entry.graphNode.admissibleHeuristicCost()
                
                //  Push each of the children onto the next level
                var index = 0
                for edge in entry.graphNode.edgeList {
                    let newCost = actualCost + edge.getCost(entry.graphNode)
                    let newEstimatedFinalCost = newCost + edge.destinationNode.admissibleHeuristicCost()
                    if (newCost < edge.destinationNode.leastCostToNode) {
                        if (newCost < bestGoalCost) {   //  Skip any that are already have higher cost than a found goal
                            var newEntry = SearchQueueEntry<T>(node: (edge.destinationNode as! T), cost: newEstimatedFinalCost)
                            newEntry.pathToNode = entry.pathToNode
                            newEntry.pathToNode.append(index)
                            _ = enqueueOrderedByCost(newEntry)  //  Sort based on estimated final cost
                            edge.destinationNode.leastCostToNode = newCost
                        }
                    }
                    index += 1
                }
                
                //  See if this is a goal node
                if (entry.graphNode === destinationNode) {
                    //  See if it is better than any other goal node we found at this depth
                    if (entry.costToNode < bestGoalCost) {
                        bestGoalCost = entry.costToNode
                        result = entry.pathToNode
                    }
                }
            }
            else {
                //  Done checking, return the result
                return result
            }
        }
    }
    
    //  Enqueue on the 'current' list with lower cost going at the end (first to pop
    func enqueueOrderedByCost(_ newItem: SearchQueueEntry<T>) -> Bool {
        var max = currentQueueList.count
        if (max == 0) {
            currentQueueList.append(newItem)
            return true
        }
        var min = 0
        
        var mid : Int
        while (min < max) {
            mid = (min + max) / 2
            if (currentQueueList[mid].costToNode > newItem.costToNode) {
                min = mid + 1;
            }
            else {
                max = mid;
            }
        }
        if (max == min) {
            if (max != currentQueueList.count && currentQueueList[max].costToNode == newItem.costToNode) {
                return false
            }
            currentQueueList.insert(newItem, at: max)
            return true
        }
        else {
            return false
        }
    }

    
    // MARK: Convert Path
    ///  Method to convert a path (an array of edge indices) to a list
    ///  of nodes traversed by the path, including the start node.  If the path is nil, the list will be nil.
    ///  If the path is empty, the list will be contain just the start node
    open func convertPathToNodeList(_ startNode: T, path: [Int]?) -> [T]? {
        //  Check for a nil path
        if let path = path {
            //  Create the node list
            var nodeList = [T]()
            
            //  Add the start node
            var currentNode = startNode
            nodeList.append(currentNode)
            
            //  Traverse the path
            for index in path {
                currentNode = currentNode.edgeList[index].destinationNode as! T
                nodeList.append(currentNode)
            }
            return nodeList
        }
        else {
            return nil
        }
    }
}
