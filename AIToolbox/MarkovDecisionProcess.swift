//
//  MarkovDecisionProcesss.swift
//  AIToolbox
//
//  Created by Kevin Coble on 3/28/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation
import Accelerate

///  Errors that MDP routines can throw
public enum MDPErrors: ErrorType {
    case FailedSolving
    case ErrorCreatingSampleSet
    case ErrorCreatingSampleTargetValues
    case ModelInputDimensionError
    case ModelOutputDimensionError
}

///  Class to solve Markov Decision Process problems
public class MDP {
    var numStates : Int         //  If discrete states
    var numActions : Int
    var discountFactor : Double
    public var convergenceLimit = 0.0000001
    
    //  Continuos state variables
    var numSamples = 10000
    var deterministicModel = true       //  If true, we don't need to sample end states
    var nonDeterministicSampleSize = 20  //  Number of samples to take if resulting model state is not deterministic
    
    
    ///  Init MDP.  Set states to 0 for continuous states
    public init(states: Int, actions: Int, discount: Double)
    {
        numStates = states
        numActions = actions
        discountFactor = discount
    }
    
    ///  Method to set the parameters for continuous state fittedValueIteration MDP's
    public func setContinousStateParameters(sampleSize: Int, deterministic: Bool, nonDetSampleSize: Int = 20)
    {
        numSamples = sampleSize
        deterministicModel = deterministic
        nonDeterministicSampleSize = nonDetSampleSize
    }
    
    ///  Method to solve using value iteration
    ///  Returns array of actions for each state
    public func valueIteration(getActions: ((fromState: Int) -> [Int]),
            getResults: ((fromState: Int, action : Int) -> [(state: Int, probability: Double)]),
            getReward: ((fromState: Int, action : Int, toState: Int) -> Double)) -> [Int]
    {
        var π = [Int](count: numStates, repeatedValue: 0)
        var V = [Double](count: numStates, repeatedValue: 0.0)
        
        var difference = convergenceLimit + 1.0
        while (difference > convergenceLimit) {    //  Go till convergence
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
                throw MDPErrors.FailedSolving
            }
        }
        
        return π
    }
    
    ///  Computes a regression model that translates the state feature mapping to V
    public func fittedValueIteration(getRandomState: (() -> [Double]),
                              getResultingState: ((fromState: [Double], action : Int) -> [Double]),
                              getReward: ((fromState: [Double], action:Int, toState: [Double]) -> Double),
                              fitModel: Regressor) throws
    {
        //  Get a set of random states
        let sample = getRandomState()
        let sampleStates = DataSet(dataType: .Regression, inputDimension: sample.count, outputDimension: 1)
        do {
            try sampleStates.addDataPoint(input: sample, output: [0])
            //  Get the rest of the random state samples
            for _ in 1..<numSamples {
                try sampleStates.addDataPoint(input: getRandomState(), output: [0])
            }
        }
        catch {
            throw MDPErrors.ErrorCreatingSampleSet
        }
        
        //  Make sure the model fits our usage of it
        if (fitModel.getInputDimension() != sample.count) {throw MDPErrors.ModelInputDimensionError}
        if (fitModel.getOutputDimension() != 1) {throw MDPErrors.ModelOutputDimensionError}
        
        //  It is recommended that the model starts with parameters as the null vector so initial inresults are 'no expected reward for each position'.
        let initParameters = [Double](count:fitModel.getParameterDimension(), repeatedValue: 0.0)
        do {
            try fitModel.setParameters(initParameters)
        }
        catch let error {
            throw error
        }
        
        var difference = convergenceLimit + 1.0
        while (difference > convergenceLimit) {    //  Go till convergence
            difference = 0.0
            
            //  Get the target value (y) for each sample
            if (deterministicModel) {
                //  If deterministic, a single sample works
                for index in 0..<numSamples {
                    var maximumReward = -Double.infinity
                    for action in 0..<numActions {
                        var state : [Double]
                        do {
                            state = try sampleStates.getInput(index)
                            let resultState = getResultingState(fromState: state, action: action)
                            let Vprime = try fitModel.predictOne(resultState)
                            let expectedReward = getReward(fromState: state, action:action, toState: resultState) + discountFactor * Vprime[0]
                            if (expectedReward > maximumReward) { maximumReward = expectedReward }
                        }
                        catch {
                            throw MDPErrors.ErrorCreatingSampleTargetValues
                        }
                    }
                    do {
                        try sampleStates.setOutput(index, newOutput: [maximumReward])
                    }
                    catch {
                        throw MDPErrors.ErrorCreatingSampleTargetValues
                    }
                }
            }
            else {
                //  If not deterministic, sample the possible result states and average
                for index in 0..<numSamples {
                    var maximumReward = -Double.infinity
                    for action in 0..<numActions {
                        var expectedReward = 0.0
                        for _ in 0..<nonDeterministicSampleSize {
                            var state : [Double]
                            do {
                                state = try sampleStates.getInput(index)
                                let resultState = getResultingState(fromState: state, action: action)
                                let Vprime = try fitModel.predictOne(resultState)
                                expectedReward += getReward(fromState: state, action:action, toState: resultState)  + discountFactor * Vprime[0]
                            }
                            catch {
                                throw MDPErrors.ErrorCreatingSampleTargetValues
                            }
                        }
                        expectedReward /= Double(nonDeterministicSampleSize)
                        if (expectedReward > maximumReward) { maximumReward = expectedReward }
                    }
                    do {
                        try sampleStates.setOutput(index, newOutput: [maximumReward])
                    }
                    catch {
                        throw MDPErrors.ErrorCreatingSampleTargetValues
                    }
                }
            }
            
            //  Use regression to get our estimation for the V function
            do {
                try fitModel.trainRegressor(sampleStates)
            }
            catch {
                throw MDPErrors.FailedSolving
            }
        }
    }
    
    ///  Once fittedValueIteration has been used, use this function to get the action for any particular state
    public func getAction(forState: [Double],
                   getResultingState: ((fromState: [Double], action : Int) -> [Double]),
                   getReward: ((fromState: [Double], action:Int, toState: [Double]) -> Double),
                   fitModel: Regressor) -> Int
    {
        var maximumReward = -Double.infinity
        var bestAction = 0
        if (deterministicModel) {
            for action in 0..<numActions {
                let resultState = getResultingState(fromState: forState, action: action)
                do {
                    let Vprime = try fitModel.predictOne(resultState)
                    let expectedReward = getReward(fromState: forState, action:action, toState: resultState) + Vprime[0]
                    if (expectedReward > maximumReward) {
                        maximumReward = expectedReward
                        bestAction = action
                    }
                }
                catch {
                    
                }
            }
        }
        else {
            for action in 0..<numActions {
                var expectedReward = 0.0
                for _ in 0..<nonDeterministicSampleSize {
                    let resultState = getResultingState(fromState: forState, action: action)
                    do {
                        let Vprime = try fitModel.predictOne(resultState)
                        expectedReward += getReward(fromState: forState, action:action, toState: resultState) +  Vprime[0]
                    }
                    catch {
                        expectedReward = 0.0    //  Error in system that should be caught elsewhere
                    }
                }
                expectedReward /= Double(nonDeterministicSampleSize)
                if (expectedReward > maximumReward) {
                    maximumReward = expectedReward
                    bestAction = action
                }
            }
        }
        
        return bestAction
    }
    
    ///  Once fittedValueIteration has been used, use this function to get the ordered list of actions, from best to worst
    public func getActionOrder(forState: [Double],
                          getResultingState: ((fromState: [Double], action : Int) -> [Double]),
                          getReward: ((fromState: [Double], action:Int, toState: [Double]) -> Double),
                          fitModel: Regressor) -> [Int]
    {
        var actionTuple : [(action: Int, expectedReward: Double)] = []
        if (deterministicModel) {
            for action in 0..<numActions {
                let resultState = getResultingState(fromState: forState, action: action)
                do {
                    let Vprime = try fitModel.predictOne(resultState)
                    let expectedReward = getReward(fromState: forState, action:action, toState: resultState) + Vprime[0]
                    actionTuple.append((action: action, expectedReward: expectedReward))
                }
                catch {
                    
                }
            }
        }
        else {
            for action in 0..<numActions {
                var expectedReward = 0.0
                for _ in 0..<nonDeterministicSampleSize {
                    let resultState = getResultingState(fromState: forState, action: action)
                    do {
                        let Vprime = try fitModel.predictOne(resultState)
                        expectedReward += getReward(fromState: forState, action:action, toState: resultState) +  Vprime[0]
                    }
                    catch {
                        expectedReward = 0.0    //  Error in system that should be caught elsewhere
                    }
                }
                expectedReward /= Double(nonDeterministicSampleSize)
                actionTuple.append((action: action, expectedReward: expectedReward))
            }
        }
        
        //  Get the ordered list of actions
        actionTuple.sortInPlace({$0.expectedReward > $1.expectedReward})
        var actionList : [Int] = []
        for tuple in actionTuple { actionList.append(tuple.action) }
        return actionList
    }
}