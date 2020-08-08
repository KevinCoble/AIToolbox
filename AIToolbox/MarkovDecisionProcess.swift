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
public enum MDPErrors: Error {
    case mdpNotSolved
    case failedSolving
    case errorCreatingSampleSet
    case errorCreatingSampleTargetValues
    case modelInputDimensionError
    case modelOutputDimensionError
    case invalidState
    case noDataForState
}

///  Structure that defines an episode for a MDP
public struct MDPEpisode {
    public let startState : Int
    public var events : [(action: Int, resultState: Int, reward: Double)]
    
    public init(startState: Int) {
        self.startState = startState
        events = []
    }
    
    public mutating func addEvent(_ event: (action: Int, resultState: Int, reward: Double))
    {
        events.append(event)
    }
}

///  Class to solve Markov Decision Process problems
open class MDP {
    var numStates : Int         //  If discrete states
    var numActions : Int
    var γ : Double
    open var convergenceLimit = 0.0000001
    
    //  Continuous state variables
    var numSamples = 10000
    var deterministicModel = true       //  If true, we don't need to sample end states
    var nonDeterministicSampleSize = 20  //  Number of samples to take if resulting model state is not deterministic
    
    //  Calculation results for discrete state/actions
    open var π : [Int]!
    open var V : [Double]!
    open var Q : [[(action: Int, count: Int, q_a: Double)]]!       //  Array of expected rewards when taking the action from the state (array sized to state size)
    var α =  0.0        //  Future weight for TD algorithms
    
    ///  Init MDP.  Set states to 0 for continuous states
    public init(states: Int, actions: Int, discount: Double)
    {
        numStates = states
        numActions = actions
        γ = discount
    }
    
    ///  Method to set the parameters for continuous state fittedValueIteration MDP's
    open func setContinousStateParameters(_ sampleSize: Int, deterministic: Bool, nonDetSampleSize: Int = 20)
    {
        numSamples = sampleSize
        deterministicModel = deterministic
        nonDeterministicSampleSize = nonDetSampleSize
    }
    
    ///  Method to solve using value iteration
    ///  Returns array of actions for each state
    open func valueIteration(_ getActions: ((_ fromState: Int) -> [Int]),
            getResults: ((_ fromState: Int, _ action : Int) -> [(state: Int, probability: Double)]),
            getReward: ((_ fromState: Int, _ action : Int, _ toState: Int) -> Double)) -> [Int]
    {
        π = [Int](repeating: 0, count: numStates)
        V = [Double](repeating: 0.0, count: numStates)
        
        var difference = convergenceLimit + 1.0
        while (difference > convergenceLimit) {    //  Go till convergence
            difference = 0.0
            //  Update each state's value
            for state in 0..<numStates {
                //  Get the maximum value for all possible actions from this state
                var maxNewValue = -Double.infinity
                let actions = getActions(state)
                if actions.count == 0 { maxNewValue = 0.0 }    //  If an end state, future value is 0
                for action in actions {
                    //  Sum the expected rewards from all possible outcomes from the action
                    var newValue = 0.0
                    let results = getResults(state, action)
                    for result in results {
                        newValue += result.probability * (getReward(state, action, result.state) + (γ * V[result.state]))
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
    open func policyIteration(_ getActions: ((_ fromState: Int) -> [Int]),
                               getResults: ((_ fromState: Int, _ action : Int) -> [(state: Int, probability: Double)]),
                               getReward: ((_ fromState: Int, _ action : Int, _ toState: Int) -> Double)) throws -> [Int]
    {
        π = [Int](repeating: -1, count: numStates)
        V = [Double](repeating: 0.0, count: numStates)
        
        var policyChanged = true
        while (policyChanged) {    //  Go till convergence
            policyChanged = false
            
            //  Set the policy to the best action given the current values
            for state in 0..<numStates {
                let actions = getActions(state)
                if (actions.count > 0) {
                    var bestAction = actions[0]
                    var bestReward = -Double.infinity
                    for action in actions {
                        //  Determine expected reward for each action
                        var expectedReward = 0.0
                        let results = getResults(state, action)
                        for result in results {
                            expectedReward += result.probability * (getReward(state, action, result.state) + (γ * V[result.state]))
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
            var matrix = [Double](repeating: 0.0, count: numStates * numStates) //  Row is state, column is resulting state
            var constants = [Double](repeating: 0.0, count: numStates)
            for state in 0..<numStates {
                matrix[state * numStates + state] = 1.0
                if π[state] >= 0 {
                    let results = getResults(state, π[state])
                    for result in results {
                        matrix[result.state * numStates + state] -= result.probability * γ
                        constants[state] += result.probability * getReward(state, π[state], result.state)
                    }
                }
            }
            var dimA = __CLPK_integer(numStates)
            var lda = dimA
            var ldb = dimA
            var colB = __CLPK_integer(1)
            var ipiv = [__CLPK_integer](repeating: 0, count: numStates)
            var info: __CLPK_integer = 0
            dgesv_(&dimA, &colB, &matrix, &lda, &ipiv, &constants, &ldb, &info)
            if (info == 0) {
                V = constants
            }
            else {
                throw MDPErrors.failedSolving
            }
        }
        
        return π
    }
    
    ///  Once valueIteration or policyIteration has been used, use this function to get the action for any particular state
    open func getAction(_ forState: Int) throws -> Int
    {
        if (π == nil) { throw MDPErrors.mdpNotSolved }
        if (forState < 0 || forState >= numStates) { throw MDPErrors.invalidState }
        return π[forState]
    }
    
    
    ///  Initialize MDP for a discrete state Monte Carlo evaluation
    open func initDiscreteStateMonteCarlo()
    {
        Q = [[(action: Int, count: Int, q_a: Double)]](repeating: [], count: numStates)
    }
    
    //  Function to update Q for a state and action, with the given reward
    func updateQ(_ fromState: Int, action: Int, reward: Double)
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
    open func evaluateMonteCarloEpisodeEveryVisit(_ episode: MDPEpisode)
    {
        //  Iterate backwards through the episode, accumulating reward and assigning to Q
        var accumulatedReward = 0.0
        for index in stride(from: (episode.events.count-1), to: 0, by: -1) {
            //  Get the reward
            accumulatedReward *= γ
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
    open func evaluateMonteCarloEpisodeFirstVisit(_ episode: MDPEpisode)
    {
        //  Find the first instance of each state in the episode
        var firstInstance  = [Int](repeating: -1, count: numStates)
        for index in 0..<episode.events.count {
            if (firstInstance[episode.events[index].resultState] < 0) {
                firstInstance[episode.events[index].resultState] = index
            }
        }
        
        //  Iterate backwards through the episode, accumulating reward and assigning to V
        var accumulatedReward = 0.0
        for index in stride(from: (episode.events.count-1), to: 0, by: -1) {
            //  Get the reward
            accumulatedReward *= γ
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
    open func initDiscreteStateTD(_ α: Double)
    {
        self.α = α
        Q = [[(action: Int, count: Int, q_a: Double)]](repeating: [], count: numStates)
        
        //  Initialize all state/action values to 0 (we don't know what states are terminal, and those must be 0, while others are arbitrary)
        for state in 0..<numStates {
            for action in 0..<numActions {
                Q[state].append((action: action, count: 1, q_a: 0.0))
            }
        }
    }
    
    ///  Evaluate an episode of a discrete state Temporal difference evaluation using 'SARSA'
    open func evaluateTDEpisodeSARSA(_ episode: MDPEpisode)
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
            Q[S][A].q_a += α * (R + (γ * Q[Sp][Ap].q_a) - Q[S][A].q_a)
            
            //  Advance S
            S = Sp
        }
    }
    
    ///  Evaluate an episode of a discrete state Temporal difference evaluation using 'Q-Learning'
    ///  Assumes actions taken during episode are 'exploratory', not just greedy
    open func evaluateTDEpisodeQLearning(_ episode: MDPEpisode)
    {
        var S = episode.startState
        for index in 0..<episode.events.count {
            let A = episode.events[index].action
            let R = episode.events[index].reward
            let Sp = episode.events[index].resultState
            
            //  Update Q
            var maxQ = -Double.infinity
            for action in 0..<numActions {
                if (Q[Sp][action].q_a > maxQ) { maxQ = Q[Sp][action].q_a }
            }
            Q[S][A].q_a += α * (R + (γ * maxQ) - Q[S][A].q_a)
            
            //  Advance S
            S = Sp
        }
    }
    
    ///  Once Monte Carlo or TD episodes have been evaluated, use this function to get the current best (greedy) action for any particular state
    open func getGreedyAction(_ forState: Int) throws -> Int
    {
        //  Validate inputs
        if (Q == nil) { throw MDPErrors.mdpNotSolved }
        if (forState < 0 || forState >= numStates) { throw MDPErrors.invalidState }
        
        if (Q[forState].count <= 0) { throw MDPErrors.noDataForState }
        
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
    open func getεGreedyAction(_ forState: Int, ε : Double) throws -> Int
    {
        if ((Double(arc4random()) / Double(RAND_MAX)) > ε) {
            //  Return a random action
            return Int(arc4random_uniform(UInt32(numActions)))
        }
        
        //  Return a greedy action
        return try getGreedyAction(forState)
    }
    
    ///  Method to generate an episode, assuming there is an internal model
    open func generateEpisode(_ getStartingState: (() -> Int),
                                getActions: ((_ fromState: Int) -> [Int]),
                                getResults: ((_ fromState: Int, _ action : Int) -> [(state: Int, probability: Double)]),
                                getReward: ((_ fromState: Int, _ action : Int, _ toState: Int) -> Double)) throws -> MDPEpisode
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
    open func generateEvent(_ fromState: Int,
                                getActions: ((_ fromState: Int) -> [Int]),
                                getResults: ((_ fromState: Int, _ action : Int) -> [(state: Int, probability: Double)]),
                                getReward: ((_ fromState: Int, _ action : Int, _ toState: Int) -> Double)) throws -> (action: Int, resultState: Int, reward: Double)!
    {
        let actions = getActions(fromState)
        if (actions.count == 0) { return nil }
        
        //  Pick an action at random
        let actionIndex = Int(arc4random_uniform(UInt32(actions.count)))
        
        //  Get the results
        let results = getResults(fromState, actions[actionIndex])
        
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
        let reward = getReward(fromState, actions[actionIndex], results[selectedResult].state)
        
        return (action: actions[actionIndex], resultState: results[selectedResult].state, reward: reward)
    }
    
    ///  Method to generate an episode, assuming there is an internal model, but actions are ε-greedy from current learning
    open func generateεEpisode(_ getStartingState: (() -> Int),
                                ε: Double,
                                getResults: ((_ fromState: Int, _ action : Int) -> [(state: Int, probability: Double)]),
                                getReward: ((_ fromState: Int, _ action : Int, _ toState: Int) -> Double)) throws -> MDPEpisode
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
    open func generateεEvent(_ fromState: Int,
                              ε: Double,
                              getResults: ((_ fromState: Int, _ action : Int) -> [(state: Int, probability: Double)]),
                              getReward: ((_ fromState: Int, _ action : Int, _ toState: Int) -> Double)) throws -> (action: Int, resultState: Int, reward: Double)!
    {
        let action = try getεGreedyAction(fromState, ε : ε)
        
        //  Get the results
        let results = getResults(fromState, action)
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
        let reward = getReward(fromState, action, results[selectedResult].state)
        
        return (action: action, resultState: results[selectedResult].state, reward: reward)
    }
    
    ///  Computes a regression model that translates the state feature mapping to V
    open func fittedValueIteration(_ getRandomState: (() -> [Double]),
                              getResultingState: ((_ fromState: [Double], _ action : Int) -> [Double]),
                              getReward: ((_ fromState: [Double], _ action:Int, _ toState: [Double]) -> Double),
                              fitModel: Regressor) throws
    {
        //  Get a set of random states
        let sample = getRandomState()
        let sampleStates = DataSet(dataType: .regression, inputDimension: sample.count, outputDimension: 1)
        do {
            try sampleStates.addDataPoint(input: sample, output: [0])
            //  Get the rest of the random state samples
            for _ in 1..<numSamples {
                try sampleStates.addDataPoint(input: getRandomState(), output: [0])
            }
        }
        catch {
            throw MDPErrors.errorCreatingSampleSet
        }
        
        //  Make sure the model fits our usage of it
        if (fitModel.getInputDimension() != sample.count) {throw MDPErrors.modelInputDimensionError}
        if (fitModel.getOutputDimension() != 1) {throw MDPErrors.modelOutputDimensionError}
        
        //  It is recommended that the model starts with parameters as the null vector so initial inresults are 'no expected reward for each position'.
        let initParameters = [Double](repeating: 0.0, count: fitModel.getParameterDimension())
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
                            let resultState = getResultingState(state, action)
                            let Vprime = try fitModel.predictOne(resultState)
                            let expectedReward = getReward(state, action, resultState) + γ * Vprime[0]
                            if (expectedReward > maximumReward) { maximumReward = expectedReward }
                        }
                        catch {
                            throw MDPErrors.errorCreatingSampleTargetValues
                        }
                    }
                    do {
                        try sampleStates.setOutput(index, newOutput: [maximumReward])
                    }
                    catch {
                        throw MDPErrors.errorCreatingSampleTargetValues
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
                                let resultState = getResultingState(state, action)
                                let Vprime = try fitModel.predictOne(resultState)
                                expectedReward += getReward(state, action, resultState)  + γ * Vprime[0]
                            }
                            catch {
                                throw MDPErrors.errorCreatingSampleTargetValues
                            }
                        }
                        expectedReward /= Double(nonDeterministicSampleSize)
                        if (expectedReward > maximumReward) { maximumReward = expectedReward }
                    }
                    do {
                        try sampleStates.setOutput(index, newOutput: [maximumReward])
                    }
                    catch {
                        throw MDPErrors.errorCreatingSampleTargetValues
                    }
                }
            }
            
            //  Use regression to get our estimation for the V function
            do {
                try fitModel.trainRegressor(sampleStates)
            }
            catch {
                throw MDPErrors.failedSolving
            }
        }
    }
    
    ///  Once fittedValueIteration has been used, use this function to get the action for any particular state
    open func getAction(_ forState: [Double],
                   getResultingState: ((_ fromState: [Double], _ action : Int) -> [Double]),
                   getReward: ((_ fromState: [Double], _ action:Int, _ toState: [Double]) -> Double),
                   fitModel: Regressor) -> Int
    {
        var maximumReward = -Double.infinity
        var bestAction = 0
        if (deterministicModel) {
            for action in 0..<numActions {
                let resultState = getResultingState(forState, action)
                do {
                    let Vprime = try fitModel.predictOne(resultState)
                    let expectedReward = getReward(forState, action, resultState) + Vprime[0]
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
                    let resultState = getResultingState(forState, action)
                    do {
                        let Vprime = try fitModel.predictOne(resultState)
                        expectedReward += getReward(forState, action, resultState) +  Vprime[0]
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
    open func getActionOrder(_ forState: [Double],
                          getResultingState: ((_ fromState: [Double], _ action : Int) -> [Double]),
                          getReward: ((_ fromState: [Double], _ action:Int, _ toState: [Double]) -> Double),
                          fitModel: Regressor) -> [Int]
    {
        var actionTuple : [(action: Int, expectedReward: Double)] = []
        if (deterministicModel) {
            for action in 0..<numActions {
                let resultState = getResultingState(forState, action)
                do {
                    let Vprime = try fitModel.predictOne(resultState)
                    let expectedReward = getReward(forState, action, resultState) + Vprime[0]
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
                    let resultState = getResultingState(forState, action)
                    do {
                        let Vprime = try fitModel.predictOne(resultState)
                        expectedReward += getReward(forState, action, resultState) +  Vprime[0]
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
        actionTuple.sort(by: {$0.expectedReward > $1.expectedReward})
        var actionList : [Int] = []
        for tuple in actionTuple { actionList.append(tuple.action) }
        return actionList
    }
    
    ///  Function to extract the current Q* values (expected reward from each state when following greedy policy)
    open func getQStar() ->[Double]
    {
        var Qstar = [Double](repeating: -Double.infinity, count: numStates)
        for state in 0..<numStates {
            for q in Q[state] {
                let q = q.q_a / Double(q.count)
                if (q > Qstar[state]) { Qstar[state] = q }
            }
        }
        
        return Qstar
    }
}
