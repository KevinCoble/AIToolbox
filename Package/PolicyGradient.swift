//
//  PolicyGradient.swift
//  AIToolbox
//
//  Created by Kevin Coble on 12/12/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation
#if os(Linux)
#else
import Accelerate
#endif

public struct PGStep {
    public var state : [Double]        //  The starting state of the step
    public var gradient : [Double]     //  The gradient between action taken and the result of the network
    public var reward : Double         //  The reward received from taking this action
    
    public init(state initState: [Double], gradient initGradient: [Double], reward initReward : Double) {
        state = initState
        gradient = initGradient
        reward = initReward
    }
}

public class PGEpisode {
    var steps : [PGStep] = []
    
    public init() {
        
    }
    
    public func addStep(newStep : PGStep) {
        steps.append(newStep)
    }
    
    open var finalReward : Double {
        get {
            var reward = 0.0
            if let lastStep = steps.last {
                reward = lastStep.reward
            }
            return reward
        }
    }
    
    open func discountRewards(discountFactor γ : Double)
    {
        var runningSum = 0.0
        for index in stride(from: steps.count-1, to: 0, by: -1) {
            runningSum = runningSum * γ + steps[index].reward
            steps[index].reward = runningSum
        }
    }
    
    ///  Train a network on the policy - assumes network used to generate policy and reward already discounted
    open func trainPolicyNetwork(network : NeuralNetwork, trainingRate : Double, weightDecay : Double)
    {
        //  Clear the weight change accumulations
        network.clearWeightChanges()
        
        //  Accumulate weight changes from the steps
        for index in 0..<steps.count {
            //  Scale the gradient by the discounted reward
            var scaledGradient = steps[index].gradient
            vDSP_vsmulD(scaledGradient, 1, &steps[index].reward, &scaledGradient, 1, vDSP_Length(scaledGradient.count))
            //  Train the network with the scaled gradient
            network.trainWithGradient(steps[index].state, gradient: scaledGradient)
        }
        
        //  Change the weights based on the accumulations
        network.updateWeights(trainingRate: trainingRate, weightDecay: weightDecay)
    }
}
