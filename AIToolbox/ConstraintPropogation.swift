//
//  ConstraintPropogation.swift
//  AIToolbox
//
//  Created by Kevin Coble on 3/6/15.
//  Copyright (c) 2015 Kevin Coble. All rights reserved.
//

import Foundation

// MARK: ConstraintProblemVariable
///  Class for assignment variable

open class ConstraintProblemVariable {
    
    open let domainSize: Int
    var possibleSettings : [Bool]
    var remainingPossibilityCount: Int
    var assignedValueIndex: Int?
    
    public init(sizeOfDomain: Int) {
        //  Store the number of settings
        self.domainSize = sizeOfDomain
        
        //  Allocate an array of the possible settings
        possibleSettings = Array(repeating: true, count: sizeOfDomain)
        
        //  Set the number of remaining possibilities
        remainingPossibilityCount = sizeOfDomain
    }
    
    open var hasNoPossibleSettings : Bool {
        get {
            return (remainingPossibilityCount == 0)
        }
    }
    
    open var isSingleton : Bool {
        get {
            return (assignedValueIndex == nil && remainingPossibilityCount == 1)
        }
    }
    
    open var assignedValue : Int? {
        get {
            return assignedValueIndex
        }
        set {
            assignedValueIndex = assignedValue
        }
    }
    
    open var smallestAllowedValue : Int? {
        get {
            for index in 0..<domainSize {
                if (possibleSettings[index]) {return index}
            }
            return nil
        }
    }
    
    open var largestAllowedValue : Int? {
        get {
            for index in 1...domainSize {
                if (possibleSettings[domainSize - index]) {return domainSize - index}
            }
            return nil
        }
    }
    
    open func reset() {
        for index in 0..<domainSize {
            possibleSettings[index] = true
            remainingPossibilityCount = domainSize
        }
    }
    
    open func removeValuePossibility(_ varValueIndex: Int) -> Bool {        //  Return true if the possibility was actually removed
        if (varValueIndex < 0 || varValueIndex >= domainSize) { return false}
        
        var result = false
        
        if (possibleSettings[varValueIndex]) {
            remainingPossibilityCount -= 1
            possibleSettings[varValueIndex] = false
            result = true
        }
        
        return result
    }
    
    open func allowValuePossibility(_ varValueIndex: Int) -> Bool {        //  Return true if the possibility was actually returned
        if (varValueIndex < 0 || varValueIndex >= domainSize) { return false }
        
        var result = false
       
        if (!possibleSettings[varValueIndex]) {
            remainingPossibilityCount += 1
            possibleSettings[varValueIndex] = true
            result = true
        }
        
        return result
    }
    
    open func assignToNextPossibleValue() ->Bool {        //  Returns false if value cannot be assigned
        if (remainingPossibilityCount == 0) {return false}
        
        if let currentAssignment = assignedValueIndex {
            if (currentAssignment == domainSize-1) {    //  We are on the last possible assignment
                assignedValueIndex = nil
                return false
            }
            if (remainingPossibilityCount == 1) {   //  We are on the only possible assignment
                assignedValueIndex = nil
                return false
            }
            for index in currentAssignment+1..<domainSize {   //  Find the next available assignment
                if (possibleSettings[index]) {
                    assignedValueIndex = index
                    return true
                }
            }
            //  No valid possibility left
            assignedValueIndex = nil
            return false
        }
        
        else {
            //  Not assigned, find the first assignment available
            for index in 0..<domainSize {
                if (possibleSettings[index]) {
                    assignedValueIndex = index
                    break
                }
            }
        }
        
        return true
    }
    
    open func assignSingleton() -> Bool {
        if (remainingPossibilityCount != 1) { return false }
        
        for index in 0..<domainSize {
            if (possibleSettings[index]) {
                assignedValueIndex = index
                return true
            }
        }
        
        return false
    }
}


// MARK:- ConstraintProblemConstraint
///  Protocol for a constraint between two nodes
public protocol ConstraintProblemConstraint {
    var isSelfConstraint: Bool { get }
    func enforceConstraint(_ graphNodeList: [ConstraintProblemNode], forNodeIndex: Int) -> [EnforcedConstraint]
}

public struct EnforcedConstraint {
    var nodeAffected: ConstraintProblemNode
    var domainIndexRemoved: Int
}


public enum StandardConstraintType {
    case cantBeSameValueInOtherNode
    case mustBeSameValueInOtherNode
    case cantBeValue
    case mustBeGreaterThanOtherNode
    case mustBeLessThanOtherNode
}

open class InternalConstraint: ConstraintProblemConstraint {
    
    let tType: StandardConstraintType
    let nIndex: Int             //  Variable set index for 'can't be value' types, else node index
    
    public init(type: StandardConstraintType, index: Int) {
        tType = type
        nIndex = index
    }
    
    open func reciprocalConstraint(_ firstNodeIndex: Int) ->InternalConstraint? {
        switch tType {
        case .cantBeSameValueInOtherNode:
            let constraint = InternalConstraint(type: .cantBeSameValueInOtherNode, index: firstNodeIndex)
            return constraint
        case .mustBeSameValueInOtherNode:
            let constraint = InternalConstraint(type: .mustBeSameValueInOtherNode, index: firstNodeIndex)
            return constraint
        case .cantBeValue:
            return nil  //  No reciprocal for this one
        case .mustBeGreaterThanOtherNode:
            let constraint = InternalConstraint(type: .mustBeLessThanOtherNode, index: firstNodeIndex)
            return constraint
        case .mustBeLessThanOtherNode:
            let constraint = InternalConstraint(type: .mustBeGreaterThanOtherNode, index: firstNodeIndex)
            return constraint
        }
    }
    
    open var isSelfConstraint: Bool {
        return (tType == .cantBeValue)
    }
    
    open func enforceConstraint(_ graphNodeList: [ConstraintProblemNode], forNodeIndex: Int) -> [EnforcedConstraint] {
        var changeList : [EnforcedConstraint] = []
        
        let variable = graphNodeList[forNodeIndex].variable
        
        switch tType {
        case  .cantBeSameValueInOtherNode:
            if let index = variable.assignedValueIndex {
                let otherNode = graphNodeList[nIndex]
                if (otherNode.variable.removeValuePossibility(index)) {
                    changeList.append(EnforcedConstraint(nodeAffected: otherNode, domainIndexRemoved: index))
                }
            }
            
        case  .mustBeSameValueInOtherNode:
            if let index = variable.assignedValueIndex {
                let otherNode = graphNodeList[nIndex]
                for otherIndex in 0..<otherNode.variable.domainSize {
                    if (otherIndex != index) {
                        if (otherNode.variable.removeValuePossibility(otherIndex)) {
                            changeList.append(EnforcedConstraint(nodeAffected: otherNode, domainIndexRemoved: otherIndex))
                        }
                    }
                }
            }
            
        case .cantBeValue:
            if (variable.removeValuePossibility(nIndex)) {
                changeList.append(EnforcedConstraint(nodeAffected: graphNodeList[forNodeIndex], domainIndexRemoved: nIndex))
            }
            
        case .mustBeGreaterThanOtherNode:
            //  Find smallest value allowed for the node
            if let smallestValue = variable.smallestAllowedValue {
                if (smallestValue > 0) {
                    let otherNode = graphNodeList[nIndex]
                    for otherIndex in 0..<smallestValue {
                        if (otherNode.variable.removeValuePossibility(otherIndex)) {
                            changeList.append(EnforcedConstraint(nodeAffected: otherNode, domainIndexRemoved: otherIndex))
                        }
                    }
                }
            }
        
        case .mustBeLessThanOtherNode:
            //  Find largest value allowed for the node
            if let largest = variable.largestAllowedValue {
                let otherNode = graphNodeList[nIndex]
                if (largest < otherNode.variable.domainSize-1) {
                    for otherIndex in largest+1..<otherNode.variable.domainSize {
                        if (otherNode.variable.removeValuePossibility(otherIndex)) {
                            changeList.append(EnforcedConstraint(nodeAffected: otherNode, domainIndexRemoved: otherIndex))
                        }
                    }
                }
            }
        }
    
        return changeList
    }
}


// MARK:- ConstraintProblemNode
///  Class for a node with a variable and a set of constraints

open class ConstraintProblemNode {
    
    let variable : ConstraintProblemVariable
    
    var constraints : [ConstraintProblemConstraint] = []
    
    var constraintsLastEnforced: [EnforcedConstraint] = []
    
    var inQueue = false
    
    open var variableIndexValue : Int? {
        get {return variable.assignedValueIndex}
    }
    
    var nodeIndex = -1
    
    public init(variableDomainSize: Int) {
        variable = ConstraintProblemVariable(sizeOfDomain: variableDomainSize)
    }
    
    func clearConstraints() {
        constraints = []
    }
    
    func addConstraint(_ constraint: ConstraintProblemConstraint) {
        constraints.append(constraint)
    }
    
    open func resetVariable() {
        variable.reset()
    }
    
    open func processSelfConstraints(_ graphNodeList: [ConstraintProblemNode]) -> Bool
    {
        for constraint in constraints {
            if (constraint.isSelfConstraint) {
                _ = constraint.enforceConstraint(graphNodeList, forNodeIndex: nodeIndex)
            }
        }
        
        //  Verify we have a non-empty domain left
        return (!variable.hasNoPossibleSettings)
    }
    
    open func clearConstraintsLastEnforced() {
        constraintsLastEnforced = []
    }
    
    open func enforceConstraints(_ graphNodeList: [ConstraintProblemNode], nodeEnforcingConstraints: ConstraintProblemNode) -> Bool {
        //  Get our assigned domain index
        if let _ = variable.assignedValueIndex {
            //  Go through each attached constraint
            for constraint in constraints {
                nodeEnforcingConstraints.constraintsLastEnforced += constraint.enforceConstraint(graphNodeList, forNodeIndex: nodeIndex)
            }
        }
        
        return true
    }
    
    open func removeConstraintsLastEnforced() {
        for constraintEnforced in constraintsLastEnforced {
            constraintEnforced.nodeAffected.resetVariableDomainIndex(constraintEnforced.domainIndexRemoved)
        }
    }
    
    open func resetVariableDomainIndex(_ resetIndex: Int) {
        _ = variable.allowValuePossibility(resetIndex)
        
        //  If we were assigned, this un-assigns the node
        variable.assignedValueIndex = nil
    }
    
    open func assignSingleton() -> Bool {
        return variable.assignSingleton()
    }
    
    func addChangedNodesToQueue(_ queue: Queue<ConstraintProblemNode>) {
        for constraintEnforced in constraintsLastEnforced {
            if (!constraintEnforced.nodeAffected.inQueue) {
                queue.enqueue(constraintEnforced.nodeAffected)
                constraintEnforced.nodeAffected.inQueue = true
            }
        }
    }
}


// MARK: -
// MARK: ConstraintProblem Class
///  Class for a constraint problem consisting of a collection of nodes.  The nodes are (sub)classes of ConstraintProblemNode.
///   Constraints are added with the problem set, or creating ConstraintProblemConstraint conforming classes and adding those
open class ConstraintProblem {
    //  Array of nodes in the list
    var graphNodeList : [ConstraintProblemNode] = []
    
    //  Empty initializer
    public init() {
        
    }
    
    ///  Method to set an array of ConstraintProblemNode (or subclass) to the problmeem graph
    open func setNodeList(_ list: [ConstraintProblemNode]) {
        graphNodeList = list
        
        //  Number each of the nodes
        var nIndex = 0
        for node in graphNodeList {
            node.nodeIndex = nIndex
            nIndex += 1
        }

    }
    
    ///  Method to clear all constraints for the problem
    open func clearConstraints() {
        for node in graphNodeList {
            node.clearConstraints()
        }
    }
    
    ///  Method to add a value constraint to a node
    open func addValueConstraintToNode(_ node: Int, invalidValue: Int) {
        let constraint = InternalConstraint(type: .cantBeValue, index: invalidValue)
        graphNodeList[node].addConstraint(constraint)
    }
   
    ///  Method to add a constraint between two nodes
    open func addConstraintOfType(_ type: StandardConstraintType,  betweenNodeIndex firstnode: Int, andNodeIndex secondNode : Int) {
        let constraint = InternalConstraint(type: type, index: secondNode)
        graphNodeList[firstnode].addConstraint(constraint)
    }
    
    ///  Method to add a set of reciprocal constraints between two nodes
    open func addReciprocalConstraintsOfType(_ type: StandardConstraintType,  betweenNodeIndex firstNode: Int, andNodeIndex secondNode : Int) {
        let constraint = InternalConstraint(type: type, index: secondNode)
        graphNodeList[firstNode].addConstraint(constraint)
        
        if let reciprocalConstraint = constraint.reciprocalConstraint(firstNode) {
            graphNodeList[secondNode].addConstraint(reciprocalConstraint)
        }
    }
    
    ///  Method to add a custom constraint
    open func addCustomConstraint(_ constraint: ConstraintProblemConstraint, toNode: Int) {
        graphNodeList[toNode].addConstraint(constraint)
    }

    
    ///  Method to attempt to solve the problem using basic forward constraint propogation
    ///  Return true if a solution was found.  The node's variables will be set with the result
    open func solveWithForwardPropogation() -> Bool {
        //  Reset the variable possibilites for each node, then process self-inflicted constraints
        for node in graphNodeList {
            node.resetVariable()
            if (!node.processSelfConstraints(graphNodeList)) { return false }  //  self-constraints left an empty domain
        }
        
        //  Start with the first possible value for node 0
        return forwardDFS(0)
    }
    
    fileprivate func forwardDFS(_ node: Int) -> Bool {   //  Return fail if backtracking
        //  Assign this node
        while (true) {
            //  Reset any previous consraint enforcements from the last assignment
            graphNodeList[node].removeConstraintsLastEnforced()
            
            //  Assign to the next value
            if (!graphNodeList[node].variable.assignToNextPossibleValue()) {
                //  Reset any previous consraint enforcements from this assignment
                graphNodeList[node].removeConstraintsLastEnforced()
                return false
            }
            
            //  Process constraints
            graphNodeList[node].clearConstraintsLastEnforced()
            if (!graphNodeList[node].enforceConstraints(graphNodeList, nodeEnforcingConstraints: graphNodeList[node])) { continue }
            
            //  If this was the last node, return true
            if (node == graphNodeList.count-1) { return true}
            
            //  Otherwise, iterate down
            if (forwardDFS(node+1)) { return true }
        }
    }
    
    
    ///  Method to attempt to solve the problem using singleton propogation
    ///  Return true if a solution was found.  The node's variables will be set with the result
    open func solveWithSingletonPropogation() -> Bool {
        //  Reset the variable possibilites for each node, then process self-inflicted constraints
        for node in graphNodeList {
            node.resetVariable()
            if (!node.processSelfConstraints(graphNodeList)) { return false }  //  self-constraints left an empty domain
        }
        
        //  Start with the first possible value for node 0
        return singletonDFS(0)
    }
    
    fileprivate func singletonDFS(_ node: Int) -> Bool {   //  Return fail if backtracking
        let nextNode = node + 1
        
        //  If the node is assigned already (from singleton propogation), just iterate down
        if (graphNodeList[node].variable.assignedValueIndex != nil) {
            //  If this was the last node, return true
            if (nextNode == graphNodeList.count) { return true}
            
            return singletonDFS(nextNode)
        }
        
        //  Assign this node
        assignIteration : while (true) {
            //  Reset any previous consraint enforcements from the last assignment
            graphNodeList[node].removeConstraintsLastEnforced()
            
            //  Assign to the next value
            if (!graphNodeList[node].variable.assignToNextPossibleValue()) {
                //  Reset any previous consraint enforcements from this assignment
                graphNodeList[node].removeConstraintsLastEnforced()
                return false
            }
            
            if (nextNode < graphNodeList.count) {       //  If last node, skip propogation as we are done
                //  Process constraints
                graphNodeList[node].clearConstraintsLastEnforced()
                if (!graphNodeList[node].enforceConstraints(graphNodeList, nodeEnforcingConstraints: graphNodeList[node])) { continue }
                
                //  Find the singletons
                let queue = Queue<ConstraintProblemNode>()
                for index in nextNode..<graphNodeList.count {
                    graphNodeList[index].inQueue = false
                    if graphNodeList[index].variable.isSingleton {
                        queue.enqueue(graphNodeList[index])
                        graphNodeList[index].inQueue = true
                    }
                }
                
                //  Propogate all singletons information to constraint nodes
                var singleton = queue.dequeue()
                while (singleton != nil) {
                    //  Assign the singleton
                    _ = singleton!.assignSingleton()
                    
                    //  Process constraints, adding the 'backtrack' list to this node
                    if (!singleton!.enforceConstraints(graphNodeList, nodeEnforcingConstraints: graphNodeList[node])) {
                        //  Backtrack
                        continue assignIteration
                    }
                    
                    //  See if there are more singletons to add to the queue
                    for index in nextNode..<graphNodeList.count {
                        if (!graphNodeList[index].inQueue && graphNodeList[index].variable.isSingleton) {
                            queue.enqueue(graphNodeList[index])
                            graphNodeList[index].inQueue = true
                        }
                    }
                    
                    singleton = queue.dequeue()
                }
            }
            
            //  If this was the last node, return true
            if (nextNode == graphNodeList.count) { return true}
            
            //  Otherwise, iterate down
            if (singletonDFS(nextNode)) { return true }
        }
    }
    
    
    ///  Method to attempt to solve the problem using full constraint propogation
    ///  Probably only useful if less-than, greater-than, or appropriate custom constraints are in the problem
    ///  Return true if a solution was found.  The node's variables will be set with the result
    open func solveWithFullPropogation() -> Bool {
        //  Reset the variable possibilites for each node, then process self-inflicted constraints
        for node in graphNodeList {
            node.resetVariable()
            if (!node.processSelfConstraints(graphNodeList)) { return false }  //  self-constraints left an empty domain
        }
        
        //  Start with the first possible value for node 0
        return fullDFS(0)
    }
    
    fileprivate func fullDFS(_ node: Int) -> Bool {   //  Return fail if backtracking
        let nextNode = node + 1
        
        //  If the node is assigned already (from constraint propogation), just iterate down
        if (graphNodeList[node].variable.assignedValueIndex != nil) {
            //  If this was the last node, return true
            if (nextNode == graphNodeList.count) { return true}
            
            return fullDFS(nextNode)
        }
        
        //  Assign this node
        assignIteration : while (true) {
            //  Reset any previous consraint enforcements from the last assignment
            graphNodeList[node].removeConstraintsLastEnforced()
            
            //  Assign to the next value
            if (!graphNodeList[node].variable.assignToNextPossibleValue()) {
                //  Reset any previous consraint enforcements from this assignment
                graphNodeList[node].removeConstraintsLastEnforced()
                return false
            }
            
            if (nextNode < graphNodeList.count) {       //  If last node, skip propogation as we are done
                //  Create the queue, reset the 'in queue' flag
                let queue = Queue<ConstraintProblemNode>()
                for index in nextNode..<graphNodeList.count {
                    graphNodeList[index].inQueue = false
                }
                
                //  Process constraints
                graphNodeList[node].clearConstraintsLastEnforced()
                if (!graphNodeList[node].enforceConstraints(graphNodeList, nodeEnforcingConstraints: graphNodeList[node])) { continue }
                
                //  Add the modified nodes to the queue
                graphNodeList[node].addChangedNodesToQueue(queue)
                
                //  Propogate changes
                var changedNode = queue.dequeue()
                while (changedNode != nil) {
                    
                    //  Process constraints, adding the 'backtrack' list to this node
                    if (!changedNode!.enforceConstraints(graphNodeList, nodeEnforcingConstraints: graphNodeList[node])) {
                        //  Backtrack
                        continue assignIteration
                    }
                    
                    //  Add the modified nodes to the queue
                    graphNodeList[node].addChangedNodesToQueue(queue)
                    
                    changedNode = queue.dequeue()
                }
            }
            
            //  If this was the last node, return true
            if (nextNode == graphNodeList.count) { return true}
            
            //  Otherwise, iterate down
            if (fullDFS(nextNode)) { return true }
        }
    }
}
