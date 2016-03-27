//
//  Tree.swift
//  AIToolbox
//
//  Created by Kevin Coble on 2/15/15.
//  Copyright (c) 2015 Kevin Coble. All rights reserved.
//

import Foundation


//  Trees are graphs with the property that no 'branches' interconnect.  Therefore, the nodes down a path are guaranteed unique

public protocol TreeNode {
    //  Method to get child node at index.  Return nil if index out of range
    func getChildNodeForIndex(index: Int) -> TreeNode?
    
    var getChildNodeList : [TreeNode] {get}
}

public class Tree<T : TreeNode> {
    //  The root of the tree
    var rootNode : T
    
    //  Initializer with root node
    public init(initRootNode: T) {
        rootNode = initRootNode
    }
    
    ///  Computed property to get the root node
    public var getRootNode : T {return rootNode}
    
    ///  Depth-first search for inclusion in tree from specified node.  Pass nil to start at root node.
    ///  Second parameter is a completion that returns a Bool from a node, indicating the node matches the search criteria
    public func containsNodeDepthFirst(startNode : T?, searchCriteria: (T) -> Bool) -> Bool {
        var currentNode : T
        if let sNode = startNode {
            currentNode = sNode
        }
        else {
            currentNode = rootNode
        }
        
        //!!  
        let nodeList = currentNode.getChildNodeList
    
        //  See if the start, or the children of start, meet the criteria
        return nodeContainsNodeDepthFirst(currentNode, searchCriteria: searchCriteria)
    }
    private func nodeContainsNodeDepthFirst(node : T, searchCriteria: (T) -> Bool) -> Bool {
        //  If this node meets the criteria, return true
        if (searchCriteria(node)) {return true}
       
        //  Iterate through each child node, and check it
        var childIndex = 0
        var child : T?
        do {
            child = node.getChildNodeForIndex(childIndex) as! T?
            if let unwrappedChild = child {
                childIndex++
                if (nodeContainsNodeDepthFirst(unwrappedChild, searchCriteria: searchCriteria)) {return true}
            }
        }
        while (child != nil)
        
        return false
    }
    
    ///  Depth-first search for path through tree from specified node to target node.  Pass nil to start at root node.
    ///  Second parameter is a completion that returns a Bool from a node, indicating the node matches the search criteria.
    ///  Return is an optional array of integers giving the child index for the path (0-based).  Example:  [1, 0, 3]  indicates second child of start node, first child of that child, fourth child from that node.
    ///  Returns empty set [] if start node matches.  Returns nil if no node found that matches
    public func pathToNodeDepthFirst(startNode : T?, searchCriteria: (T) -> Bool) -> [Int]? {
        return nil
    }
}