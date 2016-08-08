//
//  MDPTests.swift
//  AIToolbox
//
//  Created by Kevin Coble on 3/28/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import XCTest
import AIToolbox

class MDPTests: XCTestCase {
    
    //  Set up for example problem from Andrew NG CS219 Lecture 16
    //      States:     7   8   9   10
    //                  4   -   5   6
    //                  0   1   2   3
    //              states 10 and 6 are final states
    //  Actions N, W, S, E -> 0, 1, 2, 3
    //  Rewards 10 -> +1,  6-> -1:
    //     10% chance of missing direction and going to neighboring square of target.  No movement if goes off board or to 1, 1
    let mdp = MDP(states: 11, actions: 4, discount: 0.98)
    
    //  Routine to get the start state (skip 6 and 10)
    func getStartState() -> Int
    {
        var startState = Int(arc4random_uniform(9))
        if (startState > 5) { startState += 1}
        return startState
    }
    
    //  Routine to get the actions possible from each state.  Return an empty list at ending states.  This can be from a table or calculated
    func getActions(fromState: Int) -> [Int] {
        var actions : [[Int]] = [
            [0, 3],         //  State 0
            [1, 3],         //  State 1
            [0, 1, 3],      //  State 2
            [0, 1],         //  State 3
            [0, 2],         //  State 4
            [0, 2, 3],      //  State 5
            [],             //  State 6
            [2, 3],         //  State 7
            [1, 3],         //  State 8
            [1, 2, 3],      //  State 9
            []              //  State 10
        ]
        
        return actions[fromState]
    }
    
    //  Routine to get the result of taking an action from a state.  Returns an array of state/probability tuples.  This can be from a table or calculated
    func getActionResults(fromState: Int, action : Int) -> [(state: Int, probability: Double)] {
        switch fromState {
        case 0:
            if (action == 0) {
                return [(state: 4, probability: 1.0)]
            }
            else {
                return [(state: 1, probability: 1.0)]
            }
        case 1:
            if (action == 1) {
                return [(state: 0, probability: 0.9), (state: 4, probability: 0.1)]
            }
            else {
                return [(state: 2, probability: 0.9), (state: 5, probability: 0.1)]
            }
        case 2:
            if (action == 0) {
                return [(state: 5, probability: 0.9), (state: 6, probability: 0.1)]
            }
            else if (action == 1) {
                return [(state: 1, probability: 1.0)]
            }
            else {
                return [(state: 3, probability: 0.9), (state: 6, probability: 0.1)]
            }
        case 3:
            if (action == 0) {
                return [(state: 6, probability: 0.9), (state: 5, probability: 0.1)]
            }
            else {
                return [(state: 2, probability: 0.9), (state: 5, probability: 0.1)]
            }
        case 4:
            if (action == 0) {
                return [(state: 7, probability: 0.9), (state: 8, probability: 0.1)]
            }
            else {
                return [(state: 0, probability: 0.9), (state: 1, probability: 0.1)]
            }
        case 5:
            if (action == 0) {
                return [(state: 9, probability: 0.8), (state: 8, probability: 0.1), (state: 10, probability: 0.1)]
            }
            else if (action == 2) {
                return [(state: 2, probability: 0.8), (state: 1, probability: 0.1), (state: 3, probability: 0.1)]
            }
            else {
                return [(state: 6, probability: 0.8), (state: 3, probability: 0.1), (state: 10, probability: 0.1)]
            }
        case 7:
            if (action == 2) {
                return [(state: 4, probability: 1.0)]
            }
            else {
                return [(state: 8, probability: 1.0)]
            }
        case 8:
            if (action == 1) {
                return [(state: 7, probability: 0.9), (state: 4, probability: 0.1)]
            }
            else {
                return [(state: 9, probability: 0.9), (state: 5, probability: 0.1)]
            }
        case 9:
            if (action == 1) {
                return [(state: 8, probability: 1.0)]
            }
            else if (action == 2) {
                return [(state: 5, probability: 0.9), (state: 6, probability: 0.1)]
            }
            else {
                return [(state: 10, probability: 0.9), (state: 6, probability: 0.1)]
            }
        default:
            break
        }
        return []
    }
    
    //  Routine to get the reward or penalty from taking an action from a state.  Returns an value.  This can be from a table or calculated
    func getReward(fromState: Int, action : Int, toState: Int) -> Double {
        if (toState == 10) { return 1.0 }
        if (toState == 6) { return -1.0 }
        return 0.0
    }

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }

    func testValueIteration() {
        // This tests the value iteration method.
        let results = mdp.valueIteration(getActions, getResults: getActionResults, getReward: getReward)
        XCTAssert(results.count == 11, "MDP valueIteration.  Result size")
        XCTAssert(results[0] == 0, "MDP valueIteration.  Result for state 0")
        XCTAssert(results[1] == 1, "MDP valueIteration.  Result for state 1")
        XCTAssert(results[2] == 1, "MDP valueIteration.  Result for state 2")
        XCTAssert(results[3] == 1, "MDP valueIteration.  Result for state 3")
        XCTAssert(results[4] == 0, "MDP valueIteration.  Result for state 4")
        XCTAssert(results[5] == 0, "MDP valueIteration.  Result for state 5")
        XCTAssert(results[7] == 3, "MDP valueIteration.  Result for state 7")
        XCTAssert(results[8] == 3, "MDP valueIteration.  Result for state 8")
        XCTAssert(results[9] == 3, "MDP valueIteration.  Result for state 9")
    }
    
    func testPolicyIteration() {
        // This tests the policy iteration method.
        var results = [1]
        do {
            results = try mdp.policyIteration(getActions, getResults: getActionResults, getReward: getReward)
        }
        catch {
            print("Error solving MDP")
        }
        
        XCTAssert(results.count == 11, "MDP valueIteration.  Result size")
        XCTAssert(results[0] == 0, "MDP valueIteration.  Result for state 0")
        XCTAssert(results[1] == 1, "MDP valueIteration.  Result for state 1")
        XCTAssert(results[2] == 1, "MDP valueIteration.  Result for state 2")
        XCTAssert(results[3] == 1, "MDP valueIteration.  Result for state 3")
        XCTAssert(results[4] == 0, "MDP valueIteration.  Result for state 4")
        XCTAssert(results[5] == 0, "MDP valueIteration.  Result for state 5")
        XCTAssert(results[7] == 3, "MDP valueIteration.  Result for state 7")
        XCTAssert(results[8] == 3, "MDP valueIteration.  Result for state 8")
        XCTAssert(results[9] == 3, "MDP valueIteration.  Result for state 9")
    }
    
    func testMonteCarloEveryVisit() {
        //  Initialize the Monte Carlo parameters
        mdp.initDiscreteStateMonteCarlo()
        
        //  Train on 1000 episodes
        do {
            for _ in 0..<1000 {
                let episode = try mdp.generateEpisode(getStartState,
                                                      getActions: getActions,
                                                      getResults: getActionResults,
                                                      getReward: getReward)
                mdp.evaluateMonteCarloEpisodeEveryVisit(episode)
            }
        }
        catch {
            print("Error training Monte Carlo MDP")
        }
        
        //  Verify the result
        do {
            var bestAction: Int
            bestAction = try mdp.getAction(0, getActions: getActions,
                                           getResults: getActionResults)
            XCTAssert(bestAction == 0, "MDP Monte Carlo Every Visit.  Result for state 0")
            bestAction = try mdp.getAction(1, getActions: getActions,
                                           getResults: getActionResults)
            XCTAssert(bestAction == 1, "MDP Monte Carlo Every Visit.  Result for state 1")
            bestAction = try mdp.getAction(2, getActions: getActions,
                                           getResults: getActionResults)
            XCTAssert(bestAction == 1, "MDP Monte Carlo Every Visit.  Result for state 2")
            bestAction = try mdp.getAction(3, getActions: getActions,
                                           getResults: getActionResults)
            XCTAssert(bestAction == 1, "MDP Monte Carlo Every Visit.  Result for state 3")
            bestAction = try mdp.getAction(4, getActions: getActions,
                                           getResults: getActionResults)
            XCTAssert(bestAction == 0, "MDP Monte Carlo Every Visit.  Result for state 4")
            bestAction = try mdp.getAction(5, getActions: getActions,
                                           getResults: getActionResults)
            XCTAssert(bestAction == 0, "MDP Monte Carlo Every Visit.  Result for state 5")
            bestAction = try mdp.getAction(7, getActions: getActions,
                                           getResults: getActionResults)
            XCTAssert(bestAction == 3, "MDP Monte Carlo Every Visit.  Result for state 7")
            bestAction = try mdp.getAction(8, getActions: getActions,
                                           getResults: getActionResults)
            XCTAssert(bestAction == 3, "MDP Monte Carlo Every Visit.  Result for state 8")
            bestAction = try mdp.getAction(9, getActions: getActions,
                                           getResults: getActionResults)
            XCTAssert(bestAction == 3, "MDP Monte Carlo Every Visit.  Result for state 9")
        }
        catch {
            print("Error training Monte Carlo MDP")
        }
    }
    
    func testMonteCarloFirstVisit() {
        //  Initialize the Monte Carlo parameters
        mdp.initDiscreteStateMonteCarlo()
        
        //  Train on 1000 episodes
        do {
            for _ in 0..<1000 {
                let episode = try mdp.generateEpisode(getStartState,
                                                  getActions: getActions,
                                                  getResults: getActionResults,
                                                  getReward: getReward)
                mdp.evaluateMonteCarloEpisodeFirstVisit(episode)
            }
        }
        catch {
            print("Error training Monte Carlo MDP")
        }
        
        //  Verify the result
        do {
            var bestAction: Int
            bestAction = try mdp.getAction(0, getActions: getActions,
                                       getResults: getActionResults)
            XCTAssert(bestAction == 0, "MDP Monte Carlo First Visit.  Result for state 0")
            bestAction = try mdp.getAction(1, getActions: getActions,
                                           getResults: getActionResults)
            XCTAssert(bestAction == 1, "MDP Monte Carlo First Visit.  Result for state 1")
            bestAction = try mdp.getAction(2, getActions: getActions,
                                           getResults: getActionResults)
            XCTAssert(bestAction == 1, "MDP Monte Carlo First Visit.  Result for state 2")
            bestAction = try mdp.getAction(3, getActions: getActions,
                                           getResults: getActionResults)
            XCTAssert(bestAction == 1, "MDP Monte Carlo First Visit.  Result for state 3")
            bestAction = try mdp.getAction(4, getActions: getActions,
                                           getResults: getActionResults)
            XCTAssert(bestAction == 0, "MDP Monte Carlo First Visit.  Result for state 4")
            bestAction = try mdp.getAction(5, getActions: getActions,
                                           getResults: getActionResults)
            XCTAssert(bestAction == 0, "MDP Monte Carlo First Visit.  Result for state 5")
            bestAction = try mdp.getAction(7, getActions: getActions,
                                           getResults: getActionResults)
            XCTAssert(bestAction == 3, "MDP Monte Carlo First Visit.  Result for state 7")
            bestAction = try mdp.getAction(8, getActions: getActions,
                                           getResults: getActionResults)
            XCTAssert(bestAction == 3, "MDP Monte Carlo First Visit.  Result for state 8")
            bestAction = try mdp.getAction(9, getActions: getActions,
                                           getResults: getActionResults)
            XCTAssert(bestAction == 3, "MDP Monte Carlo First Visit.  Result for state 9")
        }
        catch {
            print("Error training Monte Carlo MDP")
        }
    }

    func testPerformanceExample() {
        // This is an example of a performance test case.
        self.measureBlock {
            var results = self.mdp.valueIteration(self.getActions, getResults: self.getActionResults, getReward: self.getReward)
            do {
                results = try self.mdp.policyIteration(self.getActions, getResults: self.getActionResults, getReward: self.getReward)
            }
            catch {
                print("Error solving MDP")
            }
            XCTAssert(results.count == 11, "MDP valueIteration.  Result size")
        }
    }

}
