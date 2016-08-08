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
    case MDPNotSolved
    case FailedSolving
    case ErrorCreatingSampleSet
    case ErrorCreatingSampleTargetValues
    case ModelInputDimensionError
    case ModelOutputDimensionError
    case InvalidState
}

///  Structure that defines an episode for a MDP
public struct MDPEpisode {
    public let startState : Int
    public var events : [(action: Int, resultState: Int, reward: Double)]
    
    public init(startState: Int) {
        self.startState = startState
        events = []
    }
    
    public mutating func addEvent(event: (action: Int, resultState: Int, reward: Double))
    {
        events.append(event)
    }
}

///  Class to solve Markov Decision Process problems
public class MDP {
    var numStates : Int         //  If discrete states
    var numActions : Int
    var discountFactor : Double
    public var convergenceLimit = 0.0000001
    
    //  Continuous state variables
    var numSamples = 10000
    var deterministicModel = true       //  If true, we don't need to sample end states
    var nonDeterministicSampleSize = 20  //  Number of samples to take if resulting model state is not deterministic
    
    //  Calculation results for discrete state/actions
    var π : [Int]!
    var V : [Double]!
    var sampleCount : [Int]!      //  Number of samples for state
    
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
        π = [Int](count: numStates, repeatedValue: 0)
        V = [Double](count: numStates, repeatedValue: 0.0)
        
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
        π = [Int](count: numStates, repeatedValue: -1)
        V = [Double](count: numStates, repeatedValue: 0.0)
        
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
    
    ///  Once valueIteration or policyIteration has been used, use this function to get the action for any particular state
    public func getAction(forState: Int) throws -> Int
    {
        if (π == nil) { throw MDPErrors.MDPNotSolved }
        if (forState < 0 || forState >= numStates) { throw MDPErrors.InvalidState }
        return π[forState]
    }
    
    
    ///  Initialize MDP for a discrete state Monte Carlo evaluation
    public func initDiscreteStateMonteCarlo()
    {
        V = [Double](count: numStates, repeatedValue: 0.0)
        sampleCount  = [Int](count: numStates, repeatedValue: 0)
    }
    
    ///  Evaluate an episode of a discrete state Monte Carlo evaluation using 'every-visit'
    public func evaluateMonteCarloEpisodeEveryVisit(episode: MDPEpisode)
    {
        //  Iterate backwards through the episode, accumulating reward and assigning to V
        var accumulatedReward = 0.0
        for item in episode.events.reverse() {
            //  Get the reward
            accumulatedReward *= discountFactor
            accumulatedReward += item.reward
            
            //  Update the value
            V[item.resultState] += accumulatedReward
            sampleCount[item.resultState] += 1
        }
    }
    
    ///  Evaluate an episode of a discrete state Monte Carlo evaluation using 'first-visit'
    public func evaluateMonteCarloEpisodeFirstVisit(episode: MDPEpisode)
    {
        //  Find the first instance of each state in the episode
        var firstInstance  = [Int](count: numStates, repeatedValue: -1)
        for index in 0..<episode.events.count {
            if (firstInstance[episode.events[index].resultState] < 0) {
                firstInstance[episode.events[index].resultState] = index
            }
        }
        
        //  Iterate backwards through the episode, accumulating reward and assigning to V
        var accumulatedReward = 0.0
        for item in episode.events.reverse() {
            //  Get the reward
            accumulatedReward *= discountFactor
            accumulatedReward += item.reward
            
            //  If this was the first instance of the state, update
            if (firstInstance[item.resultState] >= 0) {
                V[item.resultState] += accumulatedReward
                sampleCount[item.resultState] += 1
            }
        }
    }
    
    ///  Once Monte Carlo episodes have been evaluated, use this function to get the current best (greedy) action for any particular state
    ///  Unvisited states can be given a specified expected reward
    public func getAction(forState: Int, getActions: ((fromState: Int) -> [Int]),
                          getResults: ((fromState: Int, action : Int) -> [(state: Int, probability: Double)]),
                          unvisitedStateReward : Double = 0.0) throws -> Int
    {
        //  Validate inputs
        if (V == nil) { throw MDPErrors.MDPNotSolved }
        if (forState < 0 || forState >= numStates) { throw MDPErrors.InvalidState }
        
        //  Get the actions that can result from this state
        let actions = getActions(fromState: forState)
        
        //  Allocate evaluation values
        var expectedReward = [Double](count: actions.count, repeatedValue: 0.0)
        
        //  Get the expected reward for each action
        for index in 0..<actions.count {
            //  Get the resulting states taking this action
            let resultingStates = getResults(fromState: forState, action: actions[index])
            for result in resultingStates {
                if (sampleCount[result.state] > 0) {
                    expectedReward[index] += V[result.state] * result.probability / Double(sampleCount[result.state])
                }
                else {
                    expectedReward[index] += unvisitedStateReward * result.probability
                }
            }
        }
        
        //  Get the action that has the most expected reward
        var bestAction = 0
        var bestReward = -Double.infinity
        for index in 0..<actions.count {
            if (expectedReward[index] > bestReward) {
                bestAction = index
                bestReward = expectedReward[index]
            }
        }
        return actions[bestAction]
    }
    
    ///  Method to generate an episode, assuming there is an internal model
    public func generateEpisode(getStartingState: (() -> Int),
                                getActions: ((fromState: Int) -> [Int]),
                                getResults: ((fromState: Int, action : Int) -> [(state: Int, probability: Double)]),
                                getReward: ((fromState: Int, action : Int, toState: Int) -> Double)) throws -> MDPEpisode
    {
        //  Get the start state
        var currentState = getStartingState()
        
        //  Initialize the return struct
        var episode = MDPEpisode(startState: currentState)
        
        //  Iterate until we get to a termination state
        while true {
            let actions = getActions(fromState: currentState)
            if (actions.count == 0) { break }
            
            //  Pick an action at random
            let action = Int(arc4random_uniform(UInt32(actions.count)))
            
            //  Get the results
            let results = getResults(fromState: currentState, action: action)
            
            //  Get the result based on the probability
            let resultProbability = Double(arc4random()) / Double(RAND_MAX)
            var accumulatedProbablility = 0.0
            var selectedResult = 0
            for index in 0..<results.count {
                if (resultProbability < (accumulatedProbablility + results[index].probability)) {
                    selectedResult = index
                    break
                }
                accumulatedProbablility += results[index].probability
            }
            
            //  Get the reward
            let reward = getReward(fromState: currentState, action: action, toState: results[selectedResult].state)
            
            //  update the state
            currentState = results[selectedResult].state
            
            //  Add the move
            episode.addEvent((action: action, resultState: currentState, reward: reward))
        }
        
        return episode
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