//
//  MarkovDecisionProcess.swift
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
    case NoDataForState
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
    public var π : [Int]!
    public var V : [Double]!
    public var Q : [[(action: Int, count: Int, q_a: Double)]]!       //  Array of expected rewards when taking the action from the state (array sized to state size)
    var α =  0.0        //  Future weight for TD algorithms
    
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
        Q = [[(action: Int, count: Int, q_a: Double)]](count: numStates, repeatedValue: [])
    }
    
    //  Function to update Q for a state and action, with the given reward
    func updateQ(fromState: Int, action: Int, reward: Double)
    {
        //  Find the action in the Q array
        var index = -1
        for i in 0..<Q[fromState].count {
            if (Q[fromState][i].action == action) {
                index = i
                break
            }
        }
        
        //  If not found, add one
        if (index < 0) {
            Q[fromState].append((action: action, count: 1, q_a: reward))
        }
        
        //  If found, update
        else {
            Q[fromState][index].count += 1
            Q[fromState][index].q_a += reward
        }
    }
    
    ///  Evaluate an episode of a discrete state Monte Carlo evaluation using 'every-visit'
    public func evaluateMonteCarloEpisodeEveryVisit(episode: MDPEpisode)
    {
        //  Iterate backwards through the episode, accumulating reward and assigning to Q
        var accumulatedReward = 0.0
        for index in (episode.events.count-1).stride(to: 0, by: -1) {
            //  Get the reward
            accumulatedReward *= discountFactor
            accumulatedReward += episode.events[index].reward
            
            //  Get the start state
            var startState = episode.startState
            if (index > 0) {
                startState = episode.events[index-1].resultState
            }
            
            //  Update the value
            updateQ(startState, action: episode.events[index].action, reward: accumulatedReward)
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
        for index in (episode.events.count-1).stride(to: 0, by: -1) {
            //  Get the reward
            accumulatedReward *= discountFactor
            accumulatedReward += episode.events[index].reward
            
            //  If this was the first instance of the state, update
            if (firstInstance[episode.events[index].resultState] >= 0) {
                //  Get the start state
                var startState = episode.startState
                if (index > 0) {
                    startState = episode.events[index-1].resultState
                }
                
                //  Update the value
                updateQ(startState, action: episode.events[index].action, reward: accumulatedReward)
            }
        }
    }
    
    
    ///  Initialize MDP for a discrete state Temporal-Difference evaluation with given future-reward weighting
    public func initDiscreteStateTD(α: Double)
    {
        self.α = α
        Q = [[(action: Int, count: Int, q_a: Double)]](count: numStates, repeatedValue: [])
        
        //  Initialize all state/action values to 0 (we don't know what states are terminal, and those must be 0, while others are arbitrary)
        for state in 0..<numStates {
            for action in 0..<numActions {
                Q[state].append((action: action, count: 1, q_a: 0.0))
            }
        }
    }
    
    ///  Evaluate an episode of a discrete state Temporal difference evaluation using 'SARSA'
    public func evaluateTDEpisodeSARSA(episode: MDPEpisode)
    {
        var S = episode.startState
        for index in 0..<episode.events.count {
            //  Get SARSA
            let A = episode.events[index].action
            let R = episode.events[index].reward
            let Sp = episode.events[index].resultState
            var Ap = 0
            if (index < episode.events.count-1) {
                Ap = episode.events[index+1].action
            }
            
            //  Update Q
            Q[S][A].q_a += α * (R + (discountFactor * Q[Sp][Ap].q_a) - Q[S][A].q_a)
            
            //  Advance S
            S = Sp
        }
    }
    
    ///  Once Monte Carlo or TD episodes have been evaluated, use this function to get the current best (greedy) action for any particular state
    public func getGreedyAction(forState: Int) throws -> Int
    {
        //  Validate inputs
        if (Q == nil) { throw MDPErrors.MDPNotSolved }
        if (forState < 0 || forState >= numStates) { throw MDPErrors.InvalidState }
        
        if (Q[forState].count <= 0) { throw MDPErrors.NoDataForState }
        
        //  Get the action that has the most expected reward
        var bestAction = 0
        var bestReward = -Double.infinity
        for index in 0..<Q[forState].count {
            let expectedReward = Q[forState][index].q_a / Double(Q[forState][index].count)
            if (expectedReward > bestReward) {
                bestAction = index
                bestReward = expectedReward
            }
        }
        return Q[forState][bestAction].action
    }
    
    ///  Once Monte Carlo or TD episodes have been evaluated, use this function to get the ε-greedy action for any particular state (greedy except random ε fraction)
    public func getεGreedyAction(forState: Int, ε : Double) throws -> Int
    {
        if ((Double(arc4random()) / Double(RAND_MAX)) > ε) {
            //  Return a random action
            return Int(arc4random_uniform(UInt32(numActions)))
        }
        
        //  Return a greedy action
        return try getGreedyAction(forState)
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
            do {
                if let event = try generateEvent(currentState, getActions: getActions, getResults: getResults, getReward: getReward) {
                    //  Add the move
                    episode.addEvent(event)
                    
                    //  update the state
                    currentState = event.resultState
                }
                else {
                    break
                }
            }
            catch let error {
                throw error
            }
        }
        
        return episode
    }
    
    ///  Method to generate an event, assuming there is an internal model
    public func generateEvent(fromState: Int,
                                getActions: ((fromState: Int) -> [Int]),
                                getResults: ((fromState: Int, action : Int) -> [(state: Int, probability: Double)]),
                                getReward: ((fromState: Int, action : Int, toState: Int) -> Double)) throws -> (action: Int, resultState: Int, reward: Double)!
    {
        let actions = getActions(fromState: fromState)
        if (actions.count == 0) { return nil }
        
        //  Pick an action at random
        let actionIndex = Int(arc4random_uniform(UInt32(actions.count)))
        
        //  Get the results
        let results = getResults(fromState: fromState, action: actions[actionIndex])
        
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
        let reward = getReward(fromState: fromState, action: actions[actionIndex], toState: results[selectedResult].state)
        
        return (action: actions[actionIndex], resultState: results[selectedResult].state, reward: reward)
    }
    
    ///  Method to generate an episode, assuming there is an internal model, but actions are ε-greedy from current learning
    public func generateεEpisode(getStartingState: (() -> Int),
                                ε: Double,
                                getResults: ((fromState: Int, action : Int) -> [(state: Int, probability: Double)]),
                                getReward: ((fromState: Int, action : Int, toState: Int) -> Double)) throws -> MDPEpisode
    {
        //  Get the start state
        var currentState = getStartingState()
        
        //  Initialize the return struct
        var episode = MDPEpisode(startState: currentState)
        
        //  Iterate until we get to a termination state
        while true {
            do {
                if let event = try generateεEvent(currentState, ε: ε, getResults: getResults, getReward: getReward) {
                    //  Add the move
                    episode.addEvent(event)
                    
                    //  update the state
                    currentState = event.resultState
                }
                else {
                    break
                }
            }
            catch let error {
                throw error
            }
        }
        
        return episode
    }
    
    ///  Method to generate an event, assuming there is an internal model, but actions are ε-greedy from current learning
    public func generateεEvent(fromState: Int,
                              ε: Double,
                              getResults: ((fromState: Int, action : Int) -> [(state: Int, probability: Double)]),
                              getReward: ((fromState: Int, action : Int, toState: Int) -> Double)) throws -> (action: Int, resultState: Int, reward: Double)!
    {
        let action = try getεGreedyAction(fromState, ε : ε)
        
        //  Get the results
        let results = getResults(fromState: fromState, action: action)
        if (results.count == 0) { return nil }
        
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
        let reward = getReward(fromState: fromState, action: action, toState: results[selectedResult].state)
        
        return (action: action, resultState: results[selectedResult].state, reward: reward)
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
    
    ///  Function to extract the current Q* values (expected reward from each state when following greedy policy)
    public func getQStar() ->[Double]
    {
        var Qstar = [Double](count: numStates, repeatedValue: -Double.infinity)
        for state in 0..<numStates {
            for q in Q[state] {
                let q = q.q_a / Double(q.count)
                if (q > Qstar[state]) { Qstar[state] = q }
            }
        }
        
        return Qstar
    }
}