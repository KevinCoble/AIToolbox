//
//  AlphaBeta.swift
//  AIToolbox
//
//  Created by Kevin Coble on 2/23/15.
//  Copyright (c) 2015 Kevin Coble. All rights reserved.
//

import Foundation

///  The AlphaBetaNode protocol is used to provide move generation and static evaluation routines to your node
public protocol AlphaBetaNode {
    
    func generateMoves(_ forMaximizer: Bool) -> [AlphaBetaNode]     //  Get the nodes for each move below this node
    func staticEvaluation() -> Double                               //  Evaluate the worth of this node
}

open class AlphaBetaGraph {
    
    public init() {
    }
    
    open func startAlphaBetaWithNode(_ startNode: AlphaBetaNode, forDepth: Int, startingAsMaximizer : Bool = true) -> AlphaBetaNode? {
        //  Start the recursion
        let α = -Double.infinity
        let β = Double.infinity
        return alphaBeta(startNode, remainingDepth: forDepth, alpha: α, beta : β, maximizer: startingAsMaximizer, topLevel: true).winningNode
    }
    
    func alphaBeta(_ currentNode: AlphaBetaNode, remainingDepth: Int, alpha : Double, beta : Double,  maximizer: Bool, topLevel : Bool) -> (value: Double, winningNode: AlphaBetaNode?) {
        //  if this is a leaf node, return the static evaluation
        if (remainingDepth == 0) {
            return (value: currentNode.staticEvaluation(), winningNode: currentNode)
        }
        let nextDepth = remainingDepth - 1
        
        //  Generate the child nodes
        let children = currentNode.generateMoves(maximizer)
        
        //  If no children, return the static evaluation for this node
        if (children.count == 0) {
            return (value: currentNode.staticEvaluation(), winningNode: currentNode)
        }
        
        if (topLevel && children.count == 1) {
            //  Only one move, so we must take it - no reason to evaluate actual values
            return (value: 0.0, winningNode: children[0])
        }
        
        var winningNode : AlphaBetaNode?
        
        var α = alpha
        var β = beta
        
        //  If the maximizer, maximize the alpha, and prune with the beta
        if (maximizer) {
            var value = -Double.infinity
            
            //  Iterate through the child nodes
            for child in children {
                let childValue = alphaBeta(child, remainingDepth: nextDepth, alpha: α, beta: β, maximizer: false, topLevel: false).value
                if (childValue > value) {
                    value = childValue
                    winningNode = child
                }
                value = childValue > value ? childValue : value
                α = value > α ? value : α
                if (β <= α) {   //  β pruning
                    break
                }
            }
            
            return (value: value, winningNode: winningNode)
        }
        
            //  If the minimizer, maximize the beta, and prune with the alpha
        else {
            var value = Double.infinity
            
            //  Iterate through the child nodes
            for child in children {
                let childValue = alphaBeta(child, remainingDepth: nextDepth, alpha: α, beta: β, maximizer: true, topLevel: false).value
                if (childValue < value) {
                    value = childValue
                    winningNode = child
                }
                β = value < β ? value : β
                if (β <= α) {         //  α pruning
                    break
                }
            }
           
            return (value: value, winningNode: winningNode)
        }
    }
    
    open func startAlphaBetaConcurrentWithNode(_ startNode: AlphaBetaNode, forDepth: Int, startingAsMaximizer : Bool = true) -> AlphaBetaNode? {
        //  Start the recursion
        let α = -Double.infinity
        let β = Double.infinity
        return alphaBetaConcurrent(startNode, remainingDepth: forDepth, alpha: α, beta : β, maximizer: startingAsMaximizer, topLevel: true).winningNode
    }
    
    func alphaBetaConcurrent(_ currentNode: AlphaBetaNode, remainingDepth: Int, alpha : Double, beta : Double,  maximizer: Bool, topLevel : Bool) -> (value: Double, winningNode: AlphaBetaNode?) {
        //  if this is a leaf node, return the static evaluation
        if (remainingDepth == 0) {
            return (value: currentNode.staticEvaluation(), winningNode: currentNode)
        }
        let nextDepth = remainingDepth - 1
        
        //  Generate the child nodes
        let children = currentNode.generateMoves(maximizer)
        
        //  If no children, return the static evaluation for this node
        if (children.count == 0) {
            return (value: currentNode.staticEvaluation(), winningNode: currentNode)
        }
        
        if (topLevel && children.count == 1) {
            //  Only one move, so we must take it - no reason to evaluate actual values
            return (value: 0.0, winningNode: children[0])
        }
        
        //  Create the value array
        var childValues : [Double] = Array(repeating: 0.0, count: children.count)
        
        //  Get the concurrent queue and group
        let tQueue = DispatchQueue.global(qos: DispatchQoS.QoSClass.default)
        let  tGroup = DispatchGroup()
        
        var winningNode : AlphaBetaNode?
        
        var α = alpha
        var β = beta
        
        //  If the maximizer, maximize the alpha, and prune with the beta
        if (maximizer) {
            //  Process the first child without concurrency - the alpha-beta range returned allows pruning of the other trees
            var value = alphaBetaConcurrent(children[0], remainingDepth: nextDepth, alpha: α, beta: β, maximizer: false, topLevel: false).value
            winningNode = children[0]
            α = value > α ? value : α
            if (β <= α) {   //  β pruning
                return (value: value, winningNode: winningNode)
            }
            
            //  Iterate through the rest of the child nodes
            if (children.count > 1) {
                for index in 1..<children.count {
                    tQueue.async(group: tGroup, execute: {() -> Void in
                        childValues[index] = self.alphaBetaConcurrent(children[index], remainingDepth: nextDepth, alpha: α, beta: β, maximizer: false, topLevel: false).value
                    })
                }
                
                //  Wait for the evaluations
                tGroup.wait()
                
                //  Prune and find best
                for index in 1..<children.count {
                    if (childValues[index] > value) {
                        value = childValues[index]
                        winningNode = children[index]
                    }
                    value = childValues[index] > value ? childValues[index] : value
                    α = value > α ? value : α
                    if (β <= α) {   //  β pruning
                        break
                    }
                }
            }
            
            return (value: value, winningNode: winningNode)
        }
            
            //  If the minimizer, maximize the beta, and prune with the alpha
        else {
            //  Process the first child without concurrency - the alpha-beta range returned allows pruning of the other trees
            var value = alphaBetaConcurrent(children[0], remainingDepth: nextDepth, alpha: α, beta: β, maximizer: true, topLevel: false).value
            winningNode = children[0]
            β = value < β ? value : β
            if (β <= α) {         //  α pruning
                return (value: value, winningNode: winningNode)
            }
            
            //  Iterate through the rest of the child nodes
            if (children.count > 1) {
                for index in 1..<children.count {
                    tQueue.async(group: tGroup, execute: {() -> Void in
                        childValues[index] = self.alphaBetaConcurrent(children[index], remainingDepth: nextDepth, alpha: α, beta: β, maximizer: true, topLevel: false).value
                    })
                }
                
                //  Wait for the evaluations
                tGroup.wait()
                
                //  Prune and find best
                for index in 1..<children.count {
                    if (childValues[index] < value) {
                        value = childValues[index]
                        winningNode = children[index]
                    }
                    β = value < β ? value : β
                    if (β <= α) {         //  α pruning
                        break
                    }
                }
            }
            
            return (value: value, winningNode: winningNode)
        }
    }
}
