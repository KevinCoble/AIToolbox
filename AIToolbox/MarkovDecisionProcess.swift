//
//  MarkovDecisionProcesss.swift
//  AIToolbox
//
//  Created by Kevin Coble on 3/28/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation
import Accelerate

///  Class to solve Markov Decision Process problems
public class MDP {
    var numStates : Int
    var numActions : Int
    var discountFactor : Double
    
    public init(states: Int, actions: Int, discount: Double)
    {
        numStates = states
        numActions = actions
        discountFactor = discount
    }
    
    ///  Method to solve using value iteration
    ///  Returns array of actions for each state
    public func valueIteration(getActions: ((fromState: Int) -> [Int]),
            getResults: ((fromState: Int, action : Int) -> [(state: Int, probability: Double)]),
            getReward: ((fromState: Int, action : Int, toState: Int) -> Double)) -> [Int]
    {
        var π = [Int](count: numStates, repeatedValue: 0)
        var V = [Double](count: numStates, repeatedValue: 0.0)
        
        var difference = 9999.0
        while (difference > 0.0000001) {    //  Go till convergence
            difference = 0.0
            //  Update each state's value
            for state in 0..<numStates {
                //  Get the maximum value for all possible actions from this state
                var maxNewValue = -Double.infinity
                let actions = getActions(fromState: state)
                if actions.count == 0 { maxNewValue = 0.0 }    //  If an end state, future value is 0
                for action in actions {
                    //  Sum the expected rewards from all possible outcomes from the action
                    var newValue = 0.0
                    let results = getResults(fromState: state, action: action)
                    for result in results {
                        newValue += result.probability * (getReward(fromState: state, action: action, toState: result.state) + (discountFactor * V[result.state]))
                    }
                    
                    //  If this is the best so far, store it
                    if (newValue > maxNewValue) {
                        maxNewValue = newValue
                        π[state] = action
                    }
                }
                
                //  Accumulate difference for convergence check
                difference += fabs(V[state] - maxNewValue)
                V[state] = maxNewValue
            }
        }
        
        return π
    }
    
    
    ///  Method to solve using policy iteration
    ///  Returns array of actions for each state
    public enum MDPErrors: ErrorType { case failedSolving }
    public func policyIteration(getActions: ((fromState: Int) -> [Int]),
                               getResults: ((fromState: Int, action : Int) -> [(state: Int, probability: Double)]),
                               getReward: ((fromState: Int, action : Int, toState: Int) -> Double)) throws -> [Int]
    {
        var π = [Int](count: numStates, repeatedValue: -1)
        var V = [Double](count: numStates, repeatedValue: 0.0)
        
        var policyChanged = true
        while (policyChanged) {    //  Go till convergence
            policyChanged = false
            
            //  Set the policy to the best action given the current values
            for state in 0..<numStates {
                let actions = getActions(fromState: state)
                if (actions.count > 0) {
                    var bestAction = actions[0]
                    var bestReward = -Double.infinity
                    for action in actions {
                        //  Determine expected reward for each action
                        var expectedReward = 0.0
                        let results = getResults(fromState: state, action: action)
                        for result in results {
                            expectedReward += result.probability * (getReward(fromState: state, action: action, toState: result.state) + (discountFactor * V[result.state]))
                        }
                        if (expectedReward > bestReward) {
                            bestReward = expectedReward
                            bestAction = action
                        }
                    }
                    //  Set to the best action found
                    if (π[state] != bestAction) { policyChanged = true }
                    π[state] = bestAction
                }
            }
            
            //  Solve for the new values
            var matrix = [Double](count: numStates * numStates, repeatedValue: 0.0) //  Row is state, column is resulting state
            var constants = [Double](count: numStates, repeatedValue: 0.0)
            for state in 0..<numStates {
                matrix[state * numStates + state] = 1.0
                if π[state] >= 0 {
                    let results = getResults(fromState: state, action: π[state])
                    for result in results {
                        matrix[result.state * numStates + state] -= result.probability * discountFactor
                        constants[state] += result.probability * getReward(fromState: state, action: π[state], toState: result.state)
                    }
                }
            }
            var dimA = Int32(numStates)
            var colB = Int32(1)
            var ipiv = [Int32](count: numStates, repeatedValue: 0)
            var info: Int32 = 0
            dgesv_(&dimA, &colB, &matrix, &dimA, &ipiv, &constants, &dimA, &info)
            if (info == 0) {
                V = constants
            }
            else {
                throw MDPErrors.failedSolving
            }
        }
        
        return π
    }
}