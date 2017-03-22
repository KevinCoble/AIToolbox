//
//  LSTMNeuralNetwork.swift
//  AIToolbox
//
//  Created by Kevin Coble on 5/20/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation
#if os(Linux)
#else
import Accelerate
#endif


final class LSTMNeuralNode {
    //  Activation function
    let activation : NeuralActivationFunction
    let numInputs : Int
    let numFeedback : Int
    
    //  Weights
    let numWeights : Int        //  This includes weights from inputs and from feedback for input, forget, cell, and output
    var Wi : [Double]
    var Ui : [Double]
    var Wf : [Double]
    var Uf : [Double]
    var Wc : [Double]
    var Uc : [Double]
    var Wo : [Double]
    var Uo : [Double]

    var h : Double //  Last result calculated
    var outputHistory : [Double] //  History of output for the sequence
    var lastCellState : Double //  Last cell state calculated
    var cellStateHistory : [Double] //  History of cell state for the sequence
    var ho : Double //  Last output gate calculated
    var outputGateHistory : [Double] //  History of output gate result for the sequence
    var hc : Double
    var memoryCellHistory : [Double] //  History of cell activation result for the sequence
    var hi : Double //  Last input gate calculated
    var inputGateHistory : [Double] //  History of input gate result for the sequence
    var hf : Double //  Last forget gate calculated
    var forgetGateHistory : [Double] //  History of forget gate result for the sequence
    var 𝟃E𝟃h : Double       //  Gradient in error with respect to output of this node for this time step plus future time steps
    var 𝟃E𝟃zo : Double      //  Gradient in error with respect to weighted sum of the output gate
    var 𝟃E𝟃zi : Double      //  Gradient in error with respect to weighted sum of the input gate
    var 𝟃E𝟃zf : Double      //  Gradient in error with respect to weighted sum of the forget gate
    var 𝟃E𝟃zc : Double      //  Gradient in error with respect to weighted sum of the memory cell
    var 𝟃E𝟃cellState : Double      //  Gradient in error with respect to state of the memory cell
    var 𝟃E𝟃Wi : [Double]
    var 𝟃E𝟃Ui : [Double]
    var 𝟃E𝟃Wf : [Double]
    var 𝟃E𝟃Uf : [Double]
    var 𝟃E𝟃Wc : [Double]
    var 𝟃E𝟃Uc : [Double]
    var 𝟃E𝟃Wo : [Double]
    var 𝟃E𝟃Uo : [Double]
    var weightUpdateMethod = NeuralWeightUpdateMethod.normal
    var weightUpdateParameter : Double?      //  Decay rate for rms prop weight updates
    var WiWeightUpdateData : [Double] = []    //  Array of running average for rmsprop
    var UiWeightUpdateData : [Double] = []    //  Array of running average for rmsprop
    var WfWeightUpdateData : [Double] = []    //  Array of running average for rmsprop
    var UfWeightUpdateData : [Double] = []    //  Array of running average for rmsprop
    var WcWeightUpdateData : [Double] = []    //  Array of running average for rmsprop
    var UcWeightUpdateData : [Double] = []    //  Array of running average for rmsprop
    var WoWeightUpdateData : [Double] = []    //  Array of running average for rmsprop
    var UoWeightUpdateData : [Double] = []    //  Array of running average for rmsprop

    ///  Create the LSTM neural network node with a set activation function
    init(numInputs : Int, numFeedbacks : Int,  activationFunction: NeuralActivationFunction)
    {
        activation = activationFunction
        self.numInputs = numInputs + 1        //  Add one weight for the bias term
        self.numFeedback = numFeedbacks
        
        //  Weights
        numWeights = (self.numInputs + self.numFeedback) * 4  //  input, forget, cell and output all have weights
        Wi = []
        Ui = []
        Wf = []
        Uf = []
        Wc = []
        Uc = []
        Wo = []
        Uo = []
        
        h = 0.0
        outputHistory = []
        lastCellState = 0.0
        cellStateHistory = []
        ho = 0.0
        outputGateHistory = []
        hc = 0.0
        memoryCellHistory = []
        hi = 0.0
        inputGateHistory = []
        hf = 0.0
        forgetGateHistory = []
        
        𝟃E𝟃h = 0.0
        𝟃E𝟃zo = 0.0
        𝟃E𝟃zi = 0.0
        𝟃E𝟃zf = 0.0
        𝟃E𝟃zc = 0.0
        𝟃E𝟃cellState = 0.0
        
        𝟃E𝟃Wi = []
        𝟃E𝟃Ui = []
        𝟃E𝟃Wf = []
        𝟃E𝟃Uf = []
        𝟃E𝟃Wc = []
        𝟃E𝟃Uc = []
        𝟃E𝟃Wo = []
        𝟃E𝟃Uo = []
    }
    
    //  Initialize the weights
    func initWeights(_ startWeights: [Double]!)
    {
        if let startWeights = startWeights {
            if (startWeights.count == 1) {
                Wi = [Double](repeating: startWeights[0], count: numInputs)
                Ui = [Double](repeating: startWeights[0], count: numFeedback)
                Wf = [Double](repeating: startWeights[0], count: numInputs)
                Uf = [Double](repeating: startWeights[0], count: numFeedback)
                Wc = [Double](repeating: startWeights[0], count: numInputs)
                Uc = [Double](repeating: startWeights[0], count: numFeedback)
                Wo = [Double](repeating: startWeights[0], count: numInputs)
                Uo = [Double](repeating: startWeights[0], count: numFeedback)
            }
            else if (startWeights.count == (numInputs+numFeedback) * 4) {
                //  Full weight array, just split into the eight weight arrays
                var index = 0
                Wi = Array(startWeights[index..<index+numInputs])
                index += numInputs
                Ui = Array(startWeights[index..<index+numFeedback])
                index += numFeedback
                Wf = Array(startWeights[index..<index+numInputs])
                index += numInputs
                Uf = Array(startWeights[index..<index+numFeedback])
                index += numFeedback
                Wc = Array(startWeights[index..<index+numInputs])
                index += numInputs
                Uc = Array(startWeights[index..<index+numFeedback])
                index += numFeedback
                Wo = Array(startWeights[index..<index+numInputs])
                index += numInputs
                Uo = Array(startWeights[index..<index+numFeedback])
                index += numFeedback
            }
            else {
                //  Get the weights and bias start indices
                let numValues = startWeights.count
                var inputStart : Int
                var forgetStart : Int
                var cellStart : Int
                var outputStart : Int
                var sectionLength : Int
                if ((numValues % 4) == 0) {
                    //  Evenly divisible by 4, pass each quarter
                    sectionLength = numValues / 4
                    inputStart = 0
                    forgetStart = sectionLength
                    cellStart = sectionLength * 2
                    outputStart = sectionLength * 3
                }
                else {
                    //  Use the values for all sections
                    inputStart = 0
                    forgetStart = 0
                    cellStart = 0
                    outputStart = 0
                    sectionLength = numValues
                }
                
                Wi = []
                var index = inputStart //  Last number (if more than 1) goes into the bias weight, then repeat the initial
                for _ in 0..<numInputs-1  {
                    if (index >= sectionLength-1) { index = inputStart }      //  Wrap if necessary
                    Wi.append(startWeights[index])
                    index += 1
                }
                Wi.append(startWeights[inputStart + sectionLength -  1])     //  Add the bias term
                
                Ui = []
                for _ in 0..<numFeedback  {
                    if (index >= sectionLength-1) { index = inputStart }      //  Wrap if necessary
                    Ui.append(startWeights[index])
                    index += 1
                }
                
                index = forgetStart
                Wf = []
                for _ in 0..<numInputs-1  {
                    if (index >= sectionLength-1) { index = forgetStart }      //  Wrap if necessary
                    Wi.append(startWeights[index])
                    index += 1
                }
                Wf.append(startWeights[forgetStart + sectionLength -  1])     //  Add the bias term
                
                Uf = []
                for _ in 0..<numFeedback  {
                    if (index >= sectionLength-1) { index = forgetStart }      //  Wrap if necessary
                    Uf.append(startWeights[index])
                    index += 1
                }
                
                index = cellStart
                Wc = []
                for _ in 0..<numInputs-1  {
                    if (index >= sectionLength-1) { index = cellStart }      //  Wrap if necessary
                    Wc.append(startWeights[index])
                    index += 1
                }
                Wc.append(startWeights[cellStart + sectionLength -  1])     //  Add the bias term
                
                Uc = []
                for _ in 0..<numFeedback  {
                    if (index >= sectionLength-1) { index = cellStart }      //  Wrap if necessary
                    Uc.append(startWeights[index])
                    index += 1
                }
                
                index = outputStart
                Wo = []
                for _ in 0..<numInputs-1  {
                    if (index >= sectionLength-1) { index = outputStart }      //  Wrap if necessary
                    Wo.append(startWeights[index])
                    index += 1
                }
                Wo.append(startWeights[outputStart + sectionLength -  1])     //  Add the bias term
                
                Uo = []
                for _ in 0..<numFeedback  {
                    if (index >= sectionLength-1) { index = outputStart }      //  Wrap if necessary
                    Uo.append(startWeights[index])
                    index += 1
                }
            }
        }
        else {
            Wi = []
            for _ in 0..<numInputs-1  {
                Wi.append(Gaussian.gaussianRandom(0.0, standardDeviation: 1.0 / Double(numInputs-1)))    //  input weights - Initialize to a random number to break initial symmetry of the network, scaled to the inputs
            }
            Wi.append(Gaussian.gaussianRandom(-2.0, standardDeviation:1.0))    //  Bias weight - Initialize to a negative number to have inputs learn to feed in
            
            Ui = []
            for _ in 0..<numFeedback  {
                Ui.append(Gaussian.gaussianRandom(0.0, standardDeviation: 1.0 / Double(numFeedback)))    //  feedback weights - Initialize to a random number to break initial symmetry of the network, scaled to the inputs
            }

            Wf = []
            for _ in 0..<numInputs-1  {
                Wf.append(Gaussian.gaussianRandom(0.0, standardDeviation: 1.0 / Double(numInputs-1)))    //  input weights - Initialize to a random number to break initial symmetry of the network, scaled to the inputs
            }
            Wf.append(Gaussian.gaussianRandom(2.0, standardDeviation:1.0))    //  Bias weight - Initialize to a positive number to turn off forget (output close to 1) until it 'learns' to forget
            
            Uf = []
            for _ in 0..<numFeedback  {
                Uf.append(Gaussian.gaussianRandom(0.0, standardDeviation: 1.0 / Double(numFeedback)))    //  feedback weights - Initialize to a random number to break initial symmetry of the network, scaled to the inputs
            }
            
            Wc = []
            for _ in 0..<numInputs-1  {
                Wc.append(Gaussian.gaussianRandom(0.0, standardDeviation: 1.0 / Double(numInputs-1)))    //  input weights - Initialize to a random number to break initial symmetry of the network, scaled to the inputs
            }
            Wc.append(Gaussian.gaussianRandom(0.0, standardDeviation:1.0))    //  Bias weight - Initialize to a random number to break initial symmetry of the network
            
            Uc = []
            for _ in 0..<numFeedback  {
                Uc.append(Gaussian.gaussianRandom(0.0, standardDeviation: 1.0 / Double(numFeedback)))    //  feedback weights - Initialize to a random number to break initial symmetry of the network, scaled to the inputs
            }
            
            Wo = []
            for _ in 0..<numInputs-1  {
                Wo.append(Gaussian.gaussianRandom(0.0, standardDeviation: 1.0 / Double(numInputs-1)))    //  input weights - Initialize to a random number to break initial symmetry of the network, scaled to the inputs
            }
            Wo.append(Gaussian.gaussianRandom(-2.0, standardDeviation:1.0))    //  Bias weight - Initialize to a negative number to limit output until network learns when output is needed
            
            Uo = []
            for _ in 0..<numFeedback  {
                Uo.append(Gaussian.gaussianRandom(0.0, standardDeviation: 1.0 / Double(numFeedback)))    //  feedback weights - Initialize to a random number to break initial symmetry of the network, scaled to the inputs
            }
        }
        
        //  If rmsprop update, allocate the momentum storage array
        if (weightUpdateMethod == .rmsProp) {
            WiWeightUpdateData = [Double](repeating: 0.0, count: numInputs)
            UiWeightUpdateData = [Double](repeating: 0.0, count: numFeedback)
            WfWeightUpdateData = [Double](repeating: 0.0, count: numInputs)
            UfWeightUpdateData = [Double](repeating: 0.0, count: numFeedback)
            WcWeightUpdateData = [Double](repeating: 0.0, count: numInputs)
            UcWeightUpdateData = [Double](repeating: 0.0, count: numFeedback)
            WoWeightUpdateData = [Double](repeating: 0.0, count: numInputs)
            UoWeightUpdateData = [Double](repeating: 0.0, count: numFeedback)
        }
    }
    
    func setNeuralWeightUpdateMethod(_ method: NeuralWeightUpdateMethod, _ parameter: Double?)
    {
        weightUpdateMethod = method
        weightUpdateParameter = parameter
    }
    
    func feedForward(_ x: [Double], hPrev: [Double]) -> Double
    {
        //  Get the input gate value
        var zi = 0.0
        var sum = 0.0
        vDSP_dotprD(Wi, 1, x, 1, &zi, vDSP_Length(numInputs))
        vDSP_dotprD(Ui, 1, hPrev, 1, &sum, vDSP_Length(numFeedback))
        zi += sum
        hi = 1.0 / (1.0 + exp(-zi))
        
        //  Get the forget gate value
        var zf = 0.0
        vDSP_dotprD(Wf, 1, x, 1, &zf, vDSP_Length(numInputs))
        vDSP_dotprD(Uf, 1, hPrev, 1, &sum, vDSP_Length(numFeedback))
        zf += sum
        hf = 1.0 / (1.0 + exp(-zf))
        
        //  Get the output gate value
        var zo = 0.0
        vDSP_dotprD(Wo, 1, x, 1, &zo, vDSP_Length(numInputs))
        vDSP_dotprD(Uo, 1, hPrev, 1, &sum, vDSP_Length(numFeedback))
        zo += sum
        ho = 1.0 / (1.0 + exp(-zo))
        
        //  Get the memory cell z sumation
        var zc = 0.0
        vDSP_dotprD(Wc, 1, x, 1, &zc, vDSP_Length(numInputs))
        vDSP_dotprD(Uc, 1, hPrev, 1, &sum, vDSP_Length(numFeedback))
        zc += sum
        
        //  Use the activation function function for the nonlinearity
        switch (activation) {
        case .none:
            hc = zc
            break
        case .hyperbolicTangent:
            hc = tanh(zc)
            break
        case .sigmoidWithCrossEntropy:
            fallthrough
        case .sigmoid:
            hc = 1.0 / (1.0 + exp(-zc))
            break
        case .rectifiedLinear:
            hc = zc
            if (zc < 0) { hc = 0.0 }
            break
        case .softSign:
            hc = zc / (1.0 + abs(zc))
            break
        case .softMax:
            hc = exp(zc)
            break
        }
        
        //  Combine the forget and input gates into the cell summation
        lastCellState = lastCellState * hf + hc * hi
        
        //  Use the activation function function for the nonlinearity
        let squashedCellState = getSquashedCellState()
        
        //  Multiply the cell value by the output gate value to get the final result
        h = squashedCellState * ho
        
        return h
    }
    
    func getSquashedCellState() -> Double
    {
        
        //  Use the activation function function for the nonlinearity
        var squashedCellState : Double
        switch (activation) {
        case .none:
            squashedCellState = lastCellState
            break
        case .hyperbolicTangent:
            squashedCellState = tanh(lastCellState)
            break
        case .sigmoidWithCrossEntropy:
            fallthrough
        case .sigmoid:
            squashedCellState = 1.0 / (1.0 + exp(-lastCellState))
            break
        case .rectifiedLinear:
            squashedCellState = lastCellState
            if (lastCellState < 0) { squashedCellState = 0.0 }
            break
        case .softSign:
            squashedCellState = lastCellState / (1.0 + abs(lastCellState))
            break
        case .softMax:
            squashedCellState = exp(lastCellState)
            break
        }
        
        return squashedCellState
    }
    
    //  Get the partial derivitive of the error with respect to the weighted sum
    func getFinalNode𝟃E𝟃zs(_ 𝟃E𝟃h: Double)
    {
        //  Store 𝟃E/𝟃h, set initial future error contributions to zero, and have the hidden layer routine do the work
        self.𝟃E𝟃h = 𝟃E𝟃h
        get𝟃E𝟃zs()
    }
    
    func reset𝟃E𝟃hs()
    {
        𝟃E𝟃h = 0.0
    }
    
    func addTo𝟃E𝟃hs(_ addition: Double)
    {
        𝟃E𝟃h += addition
    }
    
    func getWeightTimes𝟃E𝟃zs(_ weightIndex: Int) ->Double
    {
        var sum = Wo[weightIndex] * 𝟃E𝟃zo
        sum += Wf[weightIndex] * 𝟃E𝟃zf
        sum += Wc[weightIndex] * 𝟃E𝟃zc
        sum += Wi[weightIndex] * 𝟃E𝟃zi
        
        return sum
    }
    
    func getFeedbackWeightTimes𝟃E𝟃zs(_ weightIndex: Int) ->Double
    {
        var sum = Uo[weightIndex] * 𝟃E𝟃zo
        sum += Uf[weightIndex] * 𝟃E𝟃zf
        sum += Uc[weightIndex] * 𝟃E𝟃zc
        sum += Ui[weightIndex] * 𝟃E𝟃zi
        
        return sum
    }
    
    func get𝟃E𝟃zs()
    {
        //  𝟃E𝟃h contains 𝟃E/𝟃h for the current time step plus all future time steps.
        
        //  h = ho * squashedCellState   -->
        //    𝟃E/𝟃zo = 𝟃E/𝟃h ⋅ 𝟃h/𝟃ho ⋅ 𝟃ho/𝟃zo = 𝟃E/𝟃h ⋅ squashedCellState ⋅ (ho - ho²)
        //    𝟃E/𝟃cellState = 𝟃E/𝟃h ⋅ 𝟃h/𝟃squashedCellState ⋅ 𝟃squashedCellState/𝟃cellState
        //              = 𝟃E/𝟃h ⋅ ho ⋅ act'(cellState) + 𝟃E_future/𝟃cellState (from previous backpropogation step)
        𝟃E𝟃zo = 𝟃E𝟃h * getSquashedCellState() * (ho - ho * ho)
        𝟃E𝟃cellState = 𝟃E𝟃h * ho * getActPrime(getSquashedCellState()) + 𝟃E𝟃cellState
        
        //  cellState = prevCellState * hf + hc * hi   -->
        //    𝟃E/𝟃zf = 𝟃E𝟃cellState ⋅ 𝟃cellState/𝟃hf ⋅ 𝟃hf/𝟃zf = 𝟃E𝟃cellState ⋅ prevCellState ⋅ (hf - hf²)
        //    𝟃E/𝟃zc = 𝟃E𝟃cellState ⋅ 𝟃cellState/𝟃hc ⋅ 𝟃hc/𝟃zc = 𝟃E𝟃cellState ⋅ hi ⋅ act'(zc)
        //    𝟃E/𝟃zi = 𝟃E𝟃cellState ⋅ 𝟃cellState/𝟃hi ⋅ 𝟃hi/𝟃zi = 𝟃E𝟃cellState ⋅ hc ⋅ (hi - hi²)
        𝟃E𝟃zf = 𝟃E𝟃cellState * getPreviousCellState() * (hf - hf * hf)
        𝟃E𝟃zc = 𝟃E𝟃cellState * hi * getActPrime(hc)
        𝟃E𝟃zi = 𝟃E𝟃cellState * hc * (hi - hi * hi)

    }
    
    func getActPrime(_ h: Double) -> Double
    {
        //  derivitive of the non-linearity: tanh' -> 1 - result^2, sigmoid -> result - result^2, rectlinear -> 0 if result<0 else 1
        var actPrime = 0.0
        switch (activation) {
        case .none:
            break
        case .hyperbolicTangent:
            actPrime = (1 - h * h)
            break
        case .sigmoidWithCrossEntropy:
            fallthrough
        case .sigmoid:
            actPrime = (h - h * h)
            break
        case .rectifiedLinear:
            actPrime = h <= 0.0 ? 0.0 : 1.0
            break
        case .softSign:
            //  Reconstitute z from h
            var z : Double
            if (h < 0) {        //  Negative z
                z = h / (1.0 + h)
                actPrime = -1.0 / ((1.0 + z) * (1.0 + z))
            }
            else {              //  Positive z
                z = h / (1.0 - h)
                actPrime = 1.0 / ((1.0 + z) * (1.0 + z))
            }
            break
        case .softMax:
            //  Should not get here - SoftMax is only valid on output layer
            break
        }
        
        return actPrime
    }

    func getPreviousCellState() -> Double
    {
        let prevValue = cellStateHistory.last
        if (prevValue == nil) { return 0.0 }
        return prevValue!
    }

    func clearWeightChanges()
    {
        𝟃E𝟃Wi = [Double](repeating: 0.0, count: numInputs)
        𝟃E𝟃Ui = [Double](repeating: 0.0, count: numFeedback)
        𝟃E𝟃Wf = [Double](repeating: 0.0, count: numInputs)
        𝟃E𝟃Uf = [Double](repeating: 0.0, count: numFeedback)
        𝟃E𝟃Wc = [Double](repeating: 0.0, count: numInputs)
        𝟃E𝟃Uc = [Double](repeating: 0.0, count: numFeedback)
        𝟃E𝟃Wo = [Double](repeating: 0.0, count: numInputs)
        𝟃E𝟃Uo = [Double](repeating: 0.0, count: numFeedback)
    }
    
    func appendWeightChanges(_ x: [Double], hPrev: [Double]) -> Double
    {
        //  Update each weight accumulation
        
        //  With 𝟃E/𝟃zo, we can get 𝟃E/𝟃Wo.  zo = Wo⋅x + Uo⋅h(t-1)).  𝟃zo/𝟃Wo = x --> 𝟃E/𝟃Wo = 𝟃E/𝟃zo ⋅ 𝟃zo/𝟃Wo = 𝟃E/𝟃zo ⋅ x
        vDSP_vsmaD(x, 1, &𝟃E𝟃zo, 𝟃E𝟃Wo, 1, &𝟃E𝟃Wo, 1, vDSP_Length(numInputs))
        //  𝟃E/𝟃Uo.  zo = Wo⋅x + Uo⋅h(t-1).  𝟃zo/𝟃Uo = h(t-1) --> 𝟃E/𝟃Uo = 𝟃E/𝟃zo ⋅ 𝟃zo/𝟃Uo = 𝟃E/𝟃zo ⋅ h(t-1)
        vDSP_vsmaD(hPrev, 1, &𝟃E𝟃zo, 𝟃E𝟃Uo, 1, &𝟃E𝟃Uo, 1, vDSP_Length(numFeedback))

        //  With 𝟃E/𝟃zi, we can get 𝟃E/𝟃Wi.  zi = Wi⋅x + Ui⋅h(t-1).  𝟃zi/𝟃Wi = x --> 𝟃E/𝟃Wi = 𝟃E/𝟃zi ⋅ 𝟃zi/𝟃Wi = 𝟃E/𝟃zi ⋅ x
        vDSP_vsmaD(x, 1, &𝟃E𝟃zi, 𝟃E𝟃Wi, 1, &𝟃E𝟃Wi, 1, vDSP_Length(numInputs))
        //  𝟃E/𝟃Ui.  i = Wi⋅x + Ui⋅h(t-1).  𝟃zi/𝟃Ui = h(t-1) --> 𝟃E/𝟃Ui = 𝟃E/𝟃zi ⋅ 𝟃zi/𝟃Ui = 𝟃E/𝟃zi ⋅ h(t-1)
        vDSP_vsmaD(hPrev, 1, &𝟃E𝟃zi, 𝟃E𝟃Ui, 1, &𝟃E𝟃Ui, 1, vDSP_Length(numFeedback))
        
        //  With 𝟃E/𝟃zf, we can get 𝟃E/𝟃Wf.  zf = Wf⋅x + Uf⋅h(t-1).  𝟃zf/𝟃Wf = x --> 𝟃E/𝟃Wf = 𝟃E/𝟃zf ⋅ 𝟃zf/𝟃Wf = 𝟃E/𝟃zf ⋅ x
        vDSP_vsmaD(x, 1, &𝟃E𝟃zf, 𝟃E𝟃Wf, 1, &𝟃E𝟃Wf, 1, vDSP_Length(numInputs))
        //  𝟃E/𝟃Uf.  f = Wf⋅x + Uf⋅h(t-1).  𝟃zf/𝟃Uf = h(t-1) --> 𝟃E/𝟃Uf = 𝟃E/𝟃zf ⋅ 𝟃zf/𝟃Uf = 𝟃E/𝟃zf ⋅ h(t-1)
        vDSP_vsmaD(hPrev, 1, &𝟃E𝟃zf, 𝟃E𝟃Uf, 1, &𝟃E𝟃Uf, 1, vDSP_Length(numFeedback))
        
        //  With 𝟃E/𝟃zc, we can get 𝟃E/𝟃Wc.  za = Wc⋅x + Uc⋅h(t-1).  𝟃za/𝟃Wa = x --> 𝟃E/𝟃Wc = 𝟃E/𝟃zc ⋅ 𝟃zc/𝟃Wc = 𝟃E/𝟃zc ⋅ x
        vDSP_vsmaD(x, 1, &𝟃E𝟃zc, 𝟃E𝟃Wc, 1, &𝟃E𝟃Wc, 1, vDSP_Length(numInputs))
        //  𝟃E/𝟃Ua.  f = Wc⋅x + Uc⋅h(t-1).  𝟃zc/𝟃Uc = h(t-1) --> 𝟃E/𝟃Uc = 𝟃E/𝟃zc ⋅ 𝟃zc/𝟃Uc = 𝟃E/𝟃zc ⋅ h(t-1)
        vDSP_vsmaD(hPrev, 1, &𝟃E𝟃zc, 𝟃E𝟃Uc, 1, &𝟃E𝟃Uc, 1, vDSP_Length(numFeedback))
        
        return h
    }
    
    func updateWeightsFromAccumulations(_ averageTrainingRate: Double)
    {
        //  Update the weights from the accumulations
        switch weightUpdateMethod {
        case .normal:
            //  weights -= accumulation * averageTrainingRate
            var η = -averageTrainingRate
            vDSP_vsmaD(𝟃E𝟃Wi, 1, &η, Wi, 1, &Wi, 1, vDSP_Length(numInputs))
            vDSP_vsmaD(𝟃E𝟃Ui, 1, &η, Ui, 1, &Ui, 1, vDSP_Length(numFeedback))
            vDSP_vsmaD(𝟃E𝟃Wf, 1, &η, Wf, 1, &Wf, 1, vDSP_Length(numInputs))
            vDSP_vsmaD(𝟃E𝟃Uf, 1, &η, Uf, 1, &Uf, 1, vDSP_Length(numFeedback))
            vDSP_vsmaD(𝟃E𝟃Wc, 1, &η, Wc, 1, &Wc, 1, vDSP_Length(numInputs))
            vDSP_vsmaD(𝟃E𝟃Uc, 1, &η, Uc, 1, &Uc, 1, vDSP_Length(numFeedback))
            vDSP_vsmaD(𝟃E𝟃Wo, 1, &η, Wo, 1, &Wo, 1, vDSP_Length(numInputs))
            vDSP_vsmaD(𝟃E𝟃Uo, 1, &η, Uo, 1, &Uo, 1, vDSP_Length(numFeedback))
        case .rmsProp:
            //  Update the rmsProp cache for Wi --> rmsprop_cache = decay_rate * rmsprop_cache + (1 - decay_rate) * gradient²
            var gradSquared = [Double](repeating: 0.0, count: numInputs)
            vDSP_vsqD(𝟃E𝟃Wi, 1, &gradSquared, 1, vDSP_Length(numInputs))  //  Get the gradient squared
            var decay = 1.0 - weightUpdateParameter!
            vDSP_vsmulD(gradSquared, 1, &decay, &gradSquared, 1, vDSP_Length(numInputs))   //  (1 - decay_rate) * gradient²
            decay = weightUpdateParameter!
            vDSP_vsmaD(WiWeightUpdateData, 1, &decay, gradSquared, 1, &WiWeightUpdateData, 1, vDSP_Length(numInputs))
            //  Update the weights --> weight += learning_rate * gradient / (sqrt(rmsprop_cache) + 1e-5)
            for i in 0..<numInputs { gradSquared[i] = sqrt(WiWeightUpdateData[i]) }      //  Re-use gradSquared for efficiency
            var small = 1.0e-05     //  Small offset to make sure we are not dividing by zero
            vDSP_vsaddD(gradSquared, 1, &small, &gradSquared, 1, vDSP_Length(numInputs))       //  (sqrt(rmsprop_cache) + 1e-5)
            var η = -averageTrainingRate     //  Needed for unsafe pointer conversion - negate for multiply-and-add vector operation
            vDSP_svdivD(&η, gradSquared, 1, &gradSquared, 1, vDSP_Length(numInputs))
            vDSP_vmaD(𝟃E𝟃Wi, 1, gradSquared, 1, Wi, 1, &Wi, 1, vDSP_Length(numInputs))
            
            //  Update the rmsProp cache for Ui --> rmsprop_cache = decay_rate * rmsprop_cache + (1 - decay_rate) * gradient²
            gradSquared = [Double](repeating: 0.0, count: numFeedback)
            vDSP_vsqD(𝟃E𝟃Ui, 1, &gradSquared, 1, vDSP_Length(numFeedback))  //  Get the gradient squared
            decay = 1.0 - weightUpdateParameter!
            vDSP_vsmulD(gradSquared, 1, &decay, &gradSquared, 1, vDSP_Length(numFeedback))   //  (1 - decay_rate) * gradient²
            decay = weightUpdateParameter!
            vDSP_vsmaD(UiWeightUpdateData, 1, &decay, gradSquared, 1, &UiWeightUpdateData, 1, vDSP_Length(numFeedback))
            //  Update the weights --> weight += learning_rate * gradient / (sqrt(rmsprop_cache) + 1e-5)
            for i in 0..<numFeedback { gradSquared[i] = sqrt(UiWeightUpdateData[i]) }      //  Re-use gradSquared for efficiency
            small = 1.0e-05     //  Small offset to make sure we are not dividing by zero
            vDSP_vsaddD(gradSquared, 1, &small, &gradSquared, 1, vDSP_Length(numFeedback))       //  (sqrt(rmsprop_cache) + 1e-5)
            η = -averageTrainingRate     //  Needed for unsafe pointer conversion - negate for multiply-and-add vector operation
            vDSP_svdivD(&η, gradSquared, 1, &gradSquared, 1, vDSP_Length(numFeedback))
            vDSP_vmaD(𝟃E𝟃Ui, 1, gradSquared, 1, Ui, 1, &Ui, 1, vDSP_Length(numFeedback))
            
            //  Update the rmsProp cache for Wf --> rmsprop_cache = decay_rate * rmsprop_cache + (1 - decay_rate) * gradient²
            gradSquared = [Double](repeating: 0.0, count: numInputs)
            vDSP_vsqD(𝟃E𝟃Wf, 1, &gradSquared, 1, vDSP_Length(numInputs))  //  Get the gradient squared
            decay = 1.0 - weightUpdateParameter!
            vDSP_vsmulD(gradSquared, 1, &decay, &gradSquared, 1, vDSP_Length(numInputs))   //  (1 - decay_rate) * gradient²
            decay = weightUpdateParameter!
            vDSP_vsmaD(WfWeightUpdateData, 1, &decay, gradSquared, 1, &WfWeightUpdateData, 1, vDSP_Length(numInputs))
            //  Update the weights --> weight += learning_rate * gradient / (sqrt(rmsprop_cache) + 1e-5)
            for i in 0..<numInputs { gradSquared[i] = sqrt(WfWeightUpdateData[i]) }      //  Re-use gradSquared for efficiency
            small = 1.0e-05     //  Small offset to make sure we are not dividing by zero
            vDSP_vsaddD(gradSquared, 1, &small, &gradSquared, 1, vDSP_Length(numInputs))       //  (sqrt(rmsprop_cache) + 1e-5)
            η = -averageTrainingRate     //  Needed for unsafe pointer conversion - negate for multiply-and-add vector operation
            vDSP_svdivD(&η, gradSquared, 1, &gradSquared, 1, vDSP_Length(numInputs))
            vDSP_vmaD(𝟃E𝟃Wf, 1, gradSquared, 1, Wf, 1, &Wf, 1, vDSP_Length(numInputs))
            
            //  Update the rmsProp cache for Uf --> rmsprop_cache = decay_rate * rmsprop_cache + (1 - decay_rate) * gradient²
            gradSquared = [Double](repeating: 0.0, count: numFeedback)
            vDSP_vsqD(𝟃E𝟃Uf, 1, &gradSquared, 1, vDSP_Length(numFeedback))  //  Get the gradient squared
            decay = 1.0 - weightUpdateParameter!
            vDSP_vsmulD(gradSquared, 1, &decay, &gradSquared, 1, vDSP_Length(numFeedback))   //  (1 - decay_rate) * gradient²
            decay = weightUpdateParameter!
            vDSP_vsmaD(UfWeightUpdateData, 1, &decay, gradSquared, 1, &UfWeightUpdateData, 1, vDSP_Length(numFeedback))
            //  Update the weights --> weight += learning_rate * gradient / (sqrt(rmsprop_cache) + 1e-5)
            for i in 0..<numFeedback { gradSquared[i] = sqrt(UfWeightUpdateData[i]) }      //  Re-use gradSquared for efficiency
            small = 1.0e-05     //  Small offset to make sure we are not dividing by zero
            vDSP_vsaddD(gradSquared, 1, &small, &gradSquared, 1, vDSP_Length(numFeedback))       //  (sqrt(rmsprop_cache) + 1e-5)
            η = -averageTrainingRate     //  Needed for unsafe pointer conversion - negate for multiply-and-add vector operation
            vDSP_svdivD(&η, gradSquared, 1, &gradSquared, 1, vDSP_Length(numFeedback))
            vDSP_vmaD(𝟃E𝟃Uf, 1, gradSquared, 1, Uf, 1, &Uf, 1, vDSP_Length(numFeedback))
            
            //  Update the rmsProp cache for Wc --> rmsprop_cache = decay_rate * rmsprop_cache + (1 - decay_rate) * gradient²
            gradSquared = [Double](repeating: 0.0, count: numInputs)
            vDSP_vsqD(𝟃E𝟃Wc, 1, &gradSquared, 1, vDSP_Length(numInputs))  //  Get the gradient squared
            decay = 1.0 - weightUpdateParameter!
            vDSP_vsmulD(gradSquared, 1, &decay, &gradSquared, 1, vDSP_Length(numInputs))   //  (1 - decay_rate) * gradient²
            decay = weightUpdateParameter!
            vDSP_vsmaD(WcWeightUpdateData, 1, &decay, gradSquared, 1, &WcWeightUpdateData, 1, vDSP_Length(numInputs))
            //  Update the weights --> weight += learning_rate * gradient / (sqrt(rmsprop_cache) + 1e-5)
            for i in 0..<numInputs { gradSquared[i] = sqrt(WcWeightUpdateData[i]) }      //  Re-use gradSquared for efficiency
            small = 1.0e-05     //  Small offset to make sure we are not dividing by zero
            vDSP_vsaddD(gradSquared, 1, &small, &gradSquared, 1, vDSP_Length(numInputs))       //  (sqrt(rmsprop_cache) + 1e-5)
            η = -averageTrainingRate     //  Needed for unsafe pointer conversion - negate for multiply-and-add vector operation
            vDSP_svdivD(&η, gradSquared, 1, &gradSquared, 1, vDSP_Length(numInputs))
            vDSP_vmaD(𝟃E𝟃Wc, 1, gradSquared, 1, Wc, 1, &Wc, 1, vDSP_Length(numInputs))
            
            //  Update the rmsProp cache for Uc --> rmsprop_cache = decay_rate * rmsprop_cache + (1 - decay_rate) * gradient²
            gradSquared = [Double](repeating: 0.0, count: numFeedback)
            vDSP_vsqD(𝟃E𝟃Uc, 1, &gradSquared, 1, vDSP_Length(numFeedback))  //  Get the gradient squared
            decay = 1.0 - weightUpdateParameter!
            vDSP_vsmulD(gradSquared, 1, &decay, &gradSquared, 1, vDSP_Length(numFeedback))   //  (1 - decay_rate) * gradient²
            decay = weightUpdateParameter!
            vDSP_vsmaD(UcWeightUpdateData, 1, &decay, gradSquared, 1, &UcWeightUpdateData, 1, vDSP_Length(numFeedback))
            //  Update the weights --> weight += learning_rate * gradient / (sqrt(rmsprop_cache) + 1e-5)
            for i in 0..<numFeedback { gradSquared[i] = sqrt(UcWeightUpdateData[i]) }      //  Re-use gradSquared for efficiency
            small = 1.0e-05     //  Small offset to make sure we are not dividing by zero
            vDSP_vsaddD(gradSquared, 1, &small, &gradSquared, 1, vDSP_Length(numFeedback))       //  (sqrt(rmsprop_cache) + 1e-5)
            η = -averageTrainingRate     //  Needed for unsafe pointer conversion - negate for multiply-and-add vector operation
            vDSP_svdivD(&η, gradSquared, 1, &gradSquared, 1, vDSP_Length(numFeedback))
            vDSP_vmaD(𝟃E𝟃Uc, 1, gradSquared, 1, Uc, 1, &Uc, 1, vDSP_Length(numFeedback))
            
            //  Update the rmsProp cache for Wo --> rmsprop_cache = decay_rate * rmsprop_cache + (1 - decay_rate) * gradient²
            gradSquared = [Double](repeating: 0.0, count: numInputs)
            vDSP_vsqD(𝟃E𝟃Wo, 1, &gradSquared, 1, vDSP_Length(numInputs))  //  Get the gradient squared
            decay = 1.0 - weightUpdateParameter!
            vDSP_vsmulD(gradSquared, 1, &decay, &gradSquared, 1, vDSP_Length(numInputs))   //  (1 - decay_rate) * gradient²
            decay = weightUpdateParameter!
            vDSP_vsmaD(WoWeightUpdateData, 1, &decay, gradSquared, 1, &WoWeightUpdateData, 1, vDSP_Length(numInputs))
            //  Update the weights --> weight += learning_rate * gradient / (sqrt(rmsprop_cache) + 1e-5)
            for i in 0..<numInputs { gradSquared[i] = sqrt(WoWeightUpdateData[i]) }      //  Re-use gradSquared for efficiency
            small = 1.0e-05     //  Small offset to make sure we are not dividing by zero
            vDSP_vsaddD(gradSquared, 1, &small, &gradSquared, 1, vDSP_Length(numInputs))       //  (sqrt(rmsprop_cache) + 1e-5)
            η = -averageTrainingRate     //  Needed for unsafe pointer conversion - negate for multiply-and-add vector operation
            vDSP_svdivD(&η, gradSquared, 1, &gradSquared, 1, vDSP_Length(numInputs))
            vDSP_vmaD(𝟃E𝟃Wo, 1, gradSquared, 1, Wo, 1, &Wo, 1, vDSP_Length(numInputs))
            
            //  Update the rmsProp cache for Uo --> rmsprop_cache = decay_rate * rmsprop_cache + (1 - decay_rate) * gradient²
            gradSquared = [Double](repeating: 0.0, count: numFeedback)
            vDSP_vsqD(𝟃E𝟃Uo, 1, &gradSquared, 1, vDSP_Length(numFeedback))  //  Get the gradient squared
            decay = 1.0 - weightUpdateParameter!
            vDSP_vsmulD(gradSquared, 1, &decay, &gradSquared, 1, vDSP_Length(numFeedback))   //  (1 - decay_rate) * gradient²
            decay = weightUpdateParameter!
            vDSP_vsmaD(UoWeightUpdateData, 1, &decay, gradSquared, 1, &UoWeightUpdateData, 1, vDSP_Length(numFeedback))
            //  Update the weights --> weight += learning_rate * gradient / (sqrt(rmsprop_cache) + 1e-5)
            for i in 0..<numFeedback { gradSquared[i] = sqrt(UoWeightUpdateData[i]) }      //  Re-use gradSquared for efficiency
            small = 1.0e-05     //  Small offset to make sure we are not dividing by zero
            vDSP_vsaddD(gradSquared, 1, &small, &gradSquared, 1, vDSP_Length(numFeedback))       //  (sqrt(rmsprop_cache) + 1e-5)
            η = -averageTrainingRate     //  Needed for unsafe pointer conversion - negate for multiply-and-add vector operation
            vDSP_svdivD(&η, gradSquared, 1, &gradSquared, 1, vDSP_Length(numFeedback))
            vDSP_vmaD(𝟃E𝟃Uo, 1, gradSquared, 1, Uo, 1, &Uo, 1, vDSP_Length(numFeedback))
        }
    }
    
    func decayWeights(_ decayFactor : Double)
    {
        var λ = decayFactor     //  Needed for unsafe pointer conversion
        vDSP_vsmulD(Wi, 1, &λ, &Wi, 1, vDSP_Length(numInputs-1))
        vDSP_vsmulD(Ui, 1, &λ, &Ui, 1, vDSP_Length(numFeedback))
        vDSP_vsmulD(Wf, 1, &λ, &Wf, 1, vDSP_Length(numInputs-1))
        vDSP_vsmulD(Uf, 1, &λ, &Uf, 1, vDSP_Length(numFeedback))
        vDSP_vsmulD(Wc, 1, &λ, &Wc, 1, vDSP_Length(numInputs-1))
        vDSP_vsmulD(Uc, 1, &λ, &Uc, 1, vDSP_Length(numFeedback))
        vDSP_vsmulD(Wo, 1, &λ, &Wo, 1, vDSP_Length(numInputs-1))
        vDSP_vsmulD(Uo, 1, &λ, &Uo, 1, vDSP_Length(numFeedback))
    }
    
    func resetSequence()
    {
        h = 0.0
        lastCellState = 0.0
        ho = 0.0
        hc = 0.0
        hi = 0.0
        hf = 0.0
        𝟃E𝟃zo = 0.0
        𝟃E𝟃zi = 0.0
        𝟃E𝟃zf = 0.0
        𝟃E𝟃zc = 0.0
        𝟃E𝟃cellState = 0.0
        outputHistory = [0.0]       //  first 'previous' value is zero
        cellStateHistory = [0.0]       //  first 'previous' value is zero
        outputGateHistory = [0.0]       //  first 'previous' value is zero
        memoryCellHistory = [0.0]       //  first 'previous' value is zero
        inputGateHistory = [0.0]       //  first 'previous' value is zero
        forgetGateHistory = [0.0]       //  first 'previous' value is zero
    }
    
    func storeRecurrentValues()
    {
        outputHistory.append(h)
        cellStateHistory.append(lastCellState)
        outputGateHistory.append(ho)
        memoryCellHistory.append(hc)
        inputGateHistory.append(hi)
        forgetGateHistory.append(hf)
    }
    
    func getLastRecurrentValue()
    {
        h = outputHistory.removeLast()
        lastCellState = cellStateHistory.removeLast()
        ho = outputGateHistory.removeLast()
        hc = memoryCellHistory.removeLast()
        hi = inputGateHistory.removeLast()
        hf = forgetGateHistory.removeLast()
    }
    
    func getPreviousOutputValue() -> Double
    {
        let prevValue = outputHistory.last
        if (prevValue == nil) { return 0.0 }
        return prevValue!
    }
}

final class LSTMNeuralLayer: NeuralLayer {
    //  Nodes
    var nodes : [LSTMNeuralNode]
    var dataSet : DataSet?              //  Sequence data set (inputs and outputs)
    
    ///  Create the neural network layer based on a tuple (number of nodes, activation function)
    init(numInputs : Int, layerDefinition: (layerType: NeuronLayerType, numNodes: Int, activation: NeuralActivationFunction, auxiliaryData: AnyObject?))
    {
        nodes = []
        for _ in 0..<layerDefinition.numNodes {
            nodes.append(LSTMNeuralNode(numInputs: numInputs, numFeedbacks: layerDefinition.numNodes, activationFunction: layerDefinition.activation))
        }
    }
    
    //  Initialize the weights
    func initWeights(_ startWeights: [Double]!)
    {
        if let startWeights = startWeights {
            if (startWeights.count >= nodes.count * nodes[0].numWeights) {
                //  If there are enough weights for all nodes, split the weights and initialize
                var startIndex = 0
                for node in nodes {
                    let subArray = Array(startWeights[startIndex...(startIndex+node.numWeights-1)])
                    node.initWeights(subArray)
                    startIndex += node.numWeights
                }
            }
            else {
                //  If there are not enough weights for all nodes, initialize each node with the set given
                for node in nodes {
                    node.initWeights(startWeights)
                }
            }
        }
        else {
            //  No specified weights - just initialize normally
            for node in nodes {
                node.initWeights(nil)
            }
        }
    }
    
    func getWeights() -> [Double]
    {
        var weights: [Double] = []
        for node in nodes {
            weights += node.Wi
            weights += node.Ui
            weights += node.Wf
            weights += node.Uf
            weights += node.Wc
            weights += node.Uc
            weights += node.Wo
            weights += node.Uo
        }
        return weights
    }
    
    func setNeuralWeightUpdateMethod(_ method: NeuralWeightUpdateMethod, _ parameter: Double?)
    {
        for node in nodes {
            node.setNeuralWeightUpdateMethod(method, parameter)
        }
    }
    
    func getLastOutput() -> [Double]
    {
        var h: [Double] = []
        for node in nodes {
            h.append(node.h)
        }
        return h
    }
   
    func getNodeCount() -> Int
    {
        return nodes.count
    }
    
    func getWeightsPerNode()-> Int
    {
        return nodes[0].numWeights
    }
    
    func getActivation()-> NeuralActivationFunction
    {
        return nodes[0].activation
    }
    
    func feedForward(_ x: [Double]) -> [Double]
    {
        //  Gather the previous outputs for the feedback
        var hPrev : [Double] = []
        for node in nodes {
            hPrev.append(node.getPreviousOutputValue())
        }
        
        var outputs : [Double] = []
        //  Assume input array already has bias constant 1.0 appended
        //  Fully-connected nodes means all nodes get the same input array
        if (nodes[0].activation == .softMax) {
            var sum = 0.0
            for node in nodes {     //  Sum each output
                sum += node.feedForward(x, hPrev: hPrev)
            }
            let scale = 1.0 / sum       //  Do division once for efficiency
            for node in nodes {     //  Get the outputs scaled by the sum to give the probability distribuition for the output
                node.h *= scale
                outputs.append(node.h)
            }
        }
        else {
            for node in nodes {
                outputs.append(node.feedForward(x, hPrev: hPrev))
            }
        }
        
        return outputs
    }
    
    func getFinalLayer𝟃E𝟃zs(_ 𝟃E𝟃h: [Double])
    {
        for nNodeIndex in 0..<nodes.count {
            //  Start with the portion from the squared error term
            nodes[nNodeIndex].getFinalNode𝟃E𝟃zs(𝟃E𝟃h[nNodeIndex])
        }
    }
    
    func getLayer𝟃E𝟃zs(_ nextLayer: NeuralLayer)
    {
        //  Get 𝟃E/𝟃h
        for nNodeIndex in 0..<nodes.count {
            //  Reset the 𝟃E/𝟃h total
            nodes[nNodeIndex].reset𝟃E𝟃hs()
            
            //  Add each portion from the nodes in the next forward layer
            nodes[nNodeIndex].addTo𝟃E𝟃hs(nextLayer.get𝟃E𝟃hForNodeInPreviousLayer(nNodeIndex))
            
            //  Add each portion from the nodes in this layer, using the feedback weights.  This adds 𝟃Efuture/𝟃h
            for node in nodes {
                nodes[nNodeIndex].addTo𝟃E𝟃hs(node.getFeedbackWeightTimes𝟃E𝟃zs(nNodeIndex))
            }
        }
        
        //  Calculate 𝟃E/𝟃zs for this time step from 𝟃E/𝟃h
        for node in nodes {
            node.get𝟃E𝟃zs()
        }
    }
    
    func get𝟃E𝟃hForNodeInPreviousLayer(_ inputIndex: Int) ->Double
    {
        var sum = 0.0
        for node in nodes {
            sum += node.getWeightTimes𝟃E𝟃zs(inputIndex)
        }
        return sum
    }
    
    func clearWeightChanges()
    {
        for node in nodes {
            node.clearWeightChanges()
        }
    }
    
    func appendWeightChanges(_ x: [Double]) -> [Double]
    {
        //  Gather the previous outputs for the feedback
        var hPrev : [Double] = []
        for node in nodes {
            hPrev.append(node.getPreviousOutputValue())
        }
        
        var outputs : [Double] = []
        //  Assume input array already has bias constant 1.0 appended
        //  Fully-connected nodes means all nodes get the same input array
        for node in nodes {
            outputs.append(node.appendWeightChanges(x, hPrev: hPrev))
        }
        
        return outputs
    }
    
    func updateWeightsFromAccumulations(_ averageTrainingRate: Double, weightDecay: Double)
    {
        //  Have each node update it's weights from the accumulations
        for node in nodes {
            if (weightDecay < 1) { node.decayWeights(weightDecay) }
            node.updateWeightsFromAccumulations(averageTrainingRate)
        }
    }
    
    func decayWeights(_ decayFactor : Double)
    {
        for node in nodes {
            node.decayWeights(decayFactor)
        }
    }
    
    func getSingleNodeClassifyValue() -> Double
    {
        let activation = nodes[0].activation
        if (activation == .hyperbolicTangent || activation == .rectifiedLinear) { return 0.0 }
        return 0.5
    }
    
    func resetSequence()
    {
        for node in nodes {
            node.resetSequence()
        }
    }
    
    func storeRecurrentValues()
    {
        for node in nodes {
            node.storeRecurrentValues()
        }
    }
    
    func retrieveRecurrentValues(_ sequenceIndex: Int)
    {
        //  Set the last recurrent value in the history array to the last output
        for node in nodes {
            node.getLastRecurrentValue()
        }
    }
    
    func gradientCheck(x: [Double], ε: Double, Δ: Double, network: NeuralNetwork)  -> Bool
    {
        //!!
        return true
    }
}
