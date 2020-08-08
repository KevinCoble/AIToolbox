//
//  MetalNeuralNetwork.swift
//  AIToolbox
//
//  Created by Kevin Coble on 1/7/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation
import Accelerate
import Metal


@available(OSX 10.11, iOS 8.0, *)
final class MetalNeuralLayer {
    
    let activationFunction: NeuralActivationFunction;
    let numInputs : Int
    let numWeights : Int
    let numNodes : Int
    
    var weights : [Float]       //  Weight array in [Node][input] order.  sized to nodes and inputs + 1 (bias term)
    var z : [Float]             //  Weighted sum for each node
    var σ : [Float]             //  Non-linear result for each node
    var delta : [Float]      //  Difference in expected output to calculated result - weighted sum from all nodes this node outputs too
    
    ///  Create the neural network layer based on a tuple (number of nodes, activation function)
    init(numInputs : Int, layerDefinition: (numNodes: Int, activation: NeuralActivationFunction))
    {
        //  Remember the definition of the layer
        activationFunction = layerDefinition.activation
        self.numInputs = numInputs
        numWeights = numInputs + 1  //  One for the bias term
        numNodes = layerDefinition.numNodes
        
        //  Create a matrix of the weights for all nodes
        weights = [Float](repeating: 0.0, count: numNodes * numWeights)
        
        //  Create the linear sum result array
        z = [Float](repeating: 0.0, count: numNodes)
        
        //  Create the non-linear result array
        σ = [Float](repeating: 0.0, count: numNodes)
        
        //  Create the delta array
        delta = [Float](repeating: 0.0, count: numNodes)
        
        //  Initialize the weights
        for index in 0..<weights.count  {
            var standardDeviation: Float = 1.0     //  Bias weights to one standard dev
            if ((index % numWeights) != numInputs) { standardDeviation = 1.0 / Float(numInputs) }
            weights[index] = Gaussian.gaussianRandomFloat(0.0, standardDeviation: standardDeviation)   //  input weights - Initialize to a random number to break initial symmetry of the network, scaled to the inputs
        }
    }
    
    
    func getLayerOutputs(_ inputs: MTLBuffer, device: MTLDevice, commandQueue: MTLCommandQueue, metalLibrary: MTLLibrary) -> MTLBuffer
    {
        //  Assume input array already has bias constant 1.0 appended
        //  Fully-connected nodes means all nodes get the same input array
        
        //  Create a buffer for storing encoded commands that are sent to GPU
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            fatalError("Cannot create command buffer.")
        }
        
        //  Create an encoder for GPU commands
        let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
        
        //  Get the name of the metal shader for the layer's non-linearity
        var programName: String
        switch (activationFunction) {
        case .none:
            programName = "sumForward"
        case .hyperbolicTangent:
            programName = "tanhForward"
        case .sigmoid:
            programName = "sigmoidForward"
        case .sigmoidWithCrossEntropy:
            programName = "sigmoidForward"
        case .rectifiedLinear:
            programName = "rectLinearForward"
        case .softSign:
            programName = "softSignForward"
        case .softMax:        //  Only valid on output (last) layer
            programName = "softMaxForward"
        }
        
        //  Set up a compute pipeline with activation function and add it to encoder
        let layerOutputProgram = metalLibrary.makeFunction(name: programName)
        var computePipelineFilter : MTLComputePipelineState?
        do {
            try computePipelineFilter = device.makeComputePipelineState(function: layerOutputProgram!)
        }
        catch {
            print("Error creating pipeline filter")
        }
        computeCommandEncoder?.setComputePipelineState(computePipelineFilter!)
        
        //  Create a MTLBuffer for the weight matrix
        let matrixByteLength = weights.count * MemoryLayout.size(ofValue: weights[0])
        let matrixBuffer = device.makeBuffer(bytes: &weights, length: matrixByteLength, options: MTLResourceOptions(rawValue: 0))
        
        //  Set the input vector for the activation function, e.g. inputs
        //    atIndex: 0 here corresponds to buffer(0) in the activation function
        computeCommandEncoder?.setBuffer(inputs, offset: 0, index: 0)
        
        //  Set the matrix vector for the activation function, e.g. matrix
        //    atIndex: 1 here corresponds to buffer(1) in the activation function
        computeCommandEncoder?.setBuffer(matrixBuffer, offset: 0, index: 1)
        
        //  Create the output vector for the summation function, e.g. zBuffer
        //    atIndex: 2 here corresponds to buffer(2) in the activation function
        let sumResultByteLength = z.count * MemoryLayout<Float>.size
        guard let zBuffer = device.makeBuffer(length: sumResultByteLength, options: MTLResourceOptions(rawValue: 0)) else {
            fatalError("zBuffer: could not make buffer")
        }
        computeCommandEncoder?.setBuffer(zBuffer, offset: 0, index: 2)
        
        //  Create the output vector for the activation function, e.g. σBuffer
        //    atIndex: 3 here corresponds to buffer(3) in the activation function
        let nextLayerInputByteLength = (σ.count + 1) * MemoryLayout<Float>.size       //  Add one for the bias term
        guard let σBuffer = device.makeBuffer(length: nextLayerInputByteLength, options: MTLResourceOptions(rawValue: 0)) else {
            fatalError("oBuffer: could not make buffer")
        }
        computeCommandEncoder?.setBuffer(σBuffer, offset: 0, index: 3)
        
        //  Create the sizing array
        let sizeArray : [Int32] = [Int32(numNodes), Int32(numWeights)]
        let sizeArrayByteLength = sizeArray.count * MemoryLayout<Int32>.size
        let sizeBuffer = device.makeBuffer(bytes: sizeArray, length: sizeArrayByteLength, options: MTLResourceOptions(rawValue: 0))
        computeCommandEncoder?.setBuffer(sizeBuffer, offset: 0, index: 4)
        
        //  Hardcoded to 32 for now (recommendation: read about threadExecutionWidth)
        let threadsPerGroup = MTLSize(width:32,height:1,depth:1)
        let numThreadgroups = MTLSize(width:((numNodes+1)+31)/32, height:1, depth:1)        //  Add one to guarantee the bias offset addition thread is performed
        computeCommandEncoder?.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        
        computeCommandEncoder?.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        //  Extract the z values
        let zData = Data(bytesNoCopy: zBuffer.contents(), count: sumResultByteLength, deallocator: .none)
        (zData as NSData).getBytes(&z, length:sumResultByteLength)
        
        //  Extract the σ values
        let activationResultByteLength = σ.count * MemoryLayout<Float>.size
        let σData = NSMutableData(bytesNoCopy: σBuffer.contents(), length: activationResultByteLength, freeWhenDone: false)
        σData.getBytes(&σ, length:activationResultByteLength)
        
        //  Add the bias term to the buffer so it is ready for the next layer
        let byteRange = NSMakeRange(activationResultByteLength, MemoryLayout<Float>.size)
        var bias : Float = 1.0
        σData.replaceBytes(in: byteRange, withBytes: &bias)
        
        
        if (activationFunction == .softMax) {
            //  Get the sum of the output nodes
            var sum: Float = 0.0;
            vDSP_vswsum(σ, 1, &sum, 1, 1, vDSP_Length(numNodes));
            
            //  Divide each element by the sum
            vDSP_vsdiv(σ, 1, &sum, &σ, 1, vDSP_Length(numNodes));
            
            //  Get the outputs into the passed-back buffer
            let byteRange = NSMakeRange(0, activationResultByteLength)
            σData.replaceBytes(in: byteRange, withBytes: &σ)
        }
        
        return σBuffer
    }
    
    //  Get the partial derivitive of the error with respect to the weighted sum
    func getFinalLayerDelta(_ expectedOutputs: [Float], device: MTLDevice, commandQueue: MTLCommandQueue, metalLibrary: MTLLibrary)
    {
        //  error = (result - expected value)^2  (squared error) - not the case for softmax or cross entropy
        //  derivitive of error = 2 * (result - expected value) * result'  (chain rule - result is a function of the sum through the non-linearity)
        //  derivitive of the non-linearity: tanh' -> 1 - result^2, sigmoid -> result - result^2, rectlinear -> 0 if result<0 else 1
        //  derivitive of error = 2 * (result - expected value) * derivitive from above

        //  Create a buffer for storing encoded commands that are sent to GPU
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            fatalError("Cannot make command buffer.")
        }
        
        //  Create an encoder for GPU commands
        let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
        
        //  Get the name of the metal shader for the layer's non-linearity
        var programName: String
        switch (activationFunction) {
        case .none:
            programName = "sumFinal"
        case .hyperbolicTangent:
            programName = "tanhFinal"
        case .sigmoid:
            programName = "sigmoidFinal"
        case .sigmoidWithCrossEntropy:
            programName = "sigmoidCrossEntropyFinal"
        case .rectifiedLinear:
            programName = "rectLinearFinal"
        case .softSign:
            programName = "softSignFinal"
        case .softMax:        //  Only valid on output (last) layer
            programName = "softMaxFinal"
        }
        
        //  Set up a compute pipeline with activation function and add it to encoder
        let layerOutputProgram = metalLibrary.makeFunction(name: programName)
        var computePipelineFilter : MTLComputePipelineState?
        do {
            try computePipelineFilter = device.makeComputePipelineState(function: layerOutputProgram!)
        }
        catch {
            print("Error creating pipeline filter")
        }
        computeCommandEncoder?.setComputePipelineState(computePipelineFilter!)
        
        //  Create a MTLBuffer for the σ vector
        let resultsByteLength = σ.count * MemoryLayout<Float>.size
        let σBuffer = device.makeBuffer(bytes: &σ, length: resultsByteLength, options: MTLResourceOptions(rawValue: 0))
        
        //  Set the result vector for the final delta function, e.g. σBuffer
        //    atIndex: 0 here corresponds to buffer(0) in the activation function
        computeCommandEncoder?.setBuffer(σBuffer, offset: 0, index: 0)
        
        //  Create a MTLBuffer for the expected results vector
        let expectedBuffer = device.makeBuffer(bytes: expectedOutputs, length: resultsByteLength, options: MTLResourceOptions(rawValue: 0))
        
        //  Set the result vector for the sigma value, e.g. expectedBuffer
        //    atIndex: 0 here corresponds to buffer(0) in the activation function
        computeCommandEncoder?.setBuffer(expectedBuffer, offset: 0, index: 1)
        
        //  Create the output vector for the final delta value, e.g. deltaBuffer
        //    atIndex: 3 here corresponds to buffer(3) in the activation function
        guard let deltaBuffer = device.makeBuffer(length: resultsByteLength, options: MTLResourceOptions(rawValue: 0)) else {
            fatalError("deltaBuffer: make buffer failed.")
        }
        computeCommandEncoder?.setBuffer(deltaBuffer, offset: 0, index: 2)
        
        //  Hardcoded to 32 for now (recommendation: read about threadExecutionWidth)
        let threadsPerGroup = MTLSize(width:32,height:1,depth:1)
        let numThreadgroups = MTLSize(width:((numNodes+1)+31)/32, height:1, depth:1)        //  Add one to guarantee the bias offset addition thread is performed
        computeCommandEncoder?.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        
        computeCommandEncoder?.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        //  Extract the delta values
        let deltaData = Data(bytesNoCopy: deltaBuffer.contents(), count: resultsByteLength, deallocator: .none)
        (deltaData as NSData).getBytes(&delta, length:resultsByteLength)
    }
    
    //  Transfer the delta from the forward layer to this one
    func getLayerDelta(_ nextLayer: MetalNeuralLayer, device: MTLDevice, commandQueue: MTLCommandQueue, metalLibrary: MTLLibrary)
    {
        //  Create a buffer for storing encoded commands that are sent to GPU
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            fatalError("Cannot make command buffer.")
        }

        //  Create an encoder for GPU commands
        let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
        
        //  Get the name of the metal shader for the layer's delta calculation
        var programName: String
        switch (activationFunction) {
        case .none:
            programName = "sumDelta"
        case .hyperbolicTangent:
            programName = "tanhDelta"
        case .sigmoid:
            programName = "sigmoidDelta"
        case .sigmoidWithCrossEntropy:
            programName = "sigmoidDelta"
        case .rectifiedLinear:
            programName = "rectLinearDelta"
        case .softSign:
            programName = "softSignDelta"
        case .softMax:        //  Only valid on output (last) layer
            print("SoftMax activation function on non-last layer")
            return
        }
        
        //  Set up a compute pipeline with activation function and add it to encoder
        let layerOutputProgram = metalLibrary.makeFunction(name: programName)
        var computePipelineFilter : MTLComputePipelineState?
        do {
            try computePipelineFilter = device.makeComputePipelineState(function: layerOutputProgram!)
        }
        catch {
            print("Error creating pipeline filter")
        }
        computeCommandEncoder?.setComputePipelineState(computePipelineFilter!)
        
        //  Create a MTLBuffer for the weight matrix from the next layer and assign to buffer 0
        let weightsByteLength = nextLayer.weights.count * MemoryLayout<Float>.size
        let weightBuffer = device.makeBuffer(bytes: &nextLayer.weights, length: weightsByteLength, options: MTLResourceOptions(rawValue: 0))
        computeCommandEncoder?.setBuffer(weightBuffer, offset: 0, index: 0)
        
        //  Create a MTLBuffer for the delta vector from the next layer and assign to buffer 1
        let deltaByteLength = nextLayer.delta.count * MemoryLayout<Float>.size
        guard let deltaBuffer = device.makeBuffer(bytes: &nextLayer.delta, length: deltaByteLength, options: MTLResourceOptions(rawValue: 0)) else {
            fatalError("deltaBuffer: make buffer failed.")
        }
        computeCommandEncoder?.setBuffer(deltaBuffer, offset: 0, index: 1)
        
        //  Create a MTLBuffer for the sizing array from the next layer and assign to buffer 2
        let sizeArray : [Int32] = [Int32(nextLayer.numNodes), Int32(numNodes)]
        let sizeArrayByteLength = sizeArray.count * MemoryLayout<Int32>.size
        let sizeBuffer = device.makeBuffer(bytes: sizeArray, length: sizeArrayByteLength, options: MTLResourceOptions(rawValue: 0))
        computeCommandEncoder?.setBuffer(sizeBuffer, offset: 0, index: 2)
        
        //  Create a MTLBuffer for the σ vector and assign to buffer 3
        let resultsByteLength = delta.count * MemoryLayout<Float>.size
        let σBuffer = device.makeBuffer(bytes: &σ, length: resultsByteLength, options: MTLResourceOptions(rawValue: 0))
        computeCommandEncoder?.setBuffer(σBuffer, offset: 0, index: 3)
        
        //  Create the output vector for the final delta function, e.g. resultBuffer and assignt to buffer 4
        //    atIndex: 3 here corresponds to buffer(3) in the activation function
        let resultBuffer = device.makeBuffer(length: resultsByteLength, options: MTLResourceOptions(rawValue: 0))
        computeCommandEncoder?.setBuffer(resultBuffer, offset: 0, index: 4)
        
        //  Hardcoded to 32 for now (recommendation: read about threadExecutionWidth)
        let threadsPerGroup = MTLSize(width:32,height:1,depth:1)
        let numThreadgroups = MTLSize(width:((numNodes+1)+31)/32, height:1, depth:1)        //  Add one to guarantee the bias offset addition thread is performed
        computeCommandEncoder?.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        
        computeCommandEncoder?.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        //  Extract the delta values
        let deltaData = Data(bytesNoCopy: deltaBuffer.contents(), count: resultsByteLength, deallocator: .none)
        (deltaData as NSData).getBytes(&delta, length:resultsByteLength)
    }
    
    func updateWeights(_ inputs: MTLBuffer, trainingRate: Float, weightDecay: Float, device: MTLDevice, commandQueue: MTLCommandQueue, metalLibrary: MTLLibrary)
    {
        //  Assume input array already has bias constant 1.0 appended
        //  Fully-connected nodes means all nodes get the same input array
        //  Create a buffer for storing encoded commands that are sent to GPU
        //  Create an encoder for GPU commands
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
            let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError("Cannot make command buffer.")
        }
        
        //  Get the name of the metal shader for the layer's weight update
        let programName = "updateWeights"
        
        //  Set up a compute pipeline with activation function and add it to encoder
        let layerOutputProgram = metalLibrary.makeFunction(name: programName)
        var computePipelineFilter : MTLComputePipelineState?
        do {
            try computePipelineFilter = device.makeComputePipelineState(function: layerOutputProgram!)
        }
        catch {
            print("Error creating pipeline filter")
        }
        computeCommandEncoder.setComputePipelineState(computePipelineFilter!)
        
        //  Set the input vector as the first buffer
        computeCommandEncoder.setBuffer(inputs, offset: 0, index: 0)
        
        //  Create a MTLBuffer for the delta vector and assign to buffer 1
        let deltaByteLength = delta.count * MemoryLayout<Float>.size
        let deltaBuffer = device.makeBuffer(bytes: &delta, length: deltaByteLength, options: MTLResourceOptions(rawValue: 0))
        computeCommandEncoder.setBuffer(deltaBuffer, offset: 0, index: 1)
        
        //  Create the parameter array
        let paramArray : [Float] = [trainingRate, weightDecay]
        let paramArrayByteLength = paramArray.count * MemoryLayout<Float>.size
        let paramBuffer = device.makeBuffer(bytes: paramArray, length: paramArrayByteLength, options: MTLResourceOptions(rawValue: 0))
        computeCommandEncoder.setBuffer(paramBuffer, offset: 0, index: 2)
        
        //  Create the sizing array
        let sizeArray : [Int32] = [Int32(numNodes), Int32(numWeights)]
        let sizeArrayByteLength = sizeArray.count * MemoryLayout<Int32>.size
        let sizeBuffer = device.makeBuffer(bytes: sizeArray, length: sizeArrayByteLength, options: MTLResourceOptions(rawValue: 0))
        computeCommandEncoder.setBuffer(sizeBuffer, offset: 0, index: 3)
        
        //  Create a MTLBuffer for the weight matrix
        let matrixByteLength = weights.count * MemoryLayout.size(ofValue: weights[0])
        guard let matrixBuffer = device.makeBuffer(bytes: &weights, length: matrixByteLength, options: MTLResourceOptions(rawValue: 0)) else {
            fatalError("matrixBuffer: make buffer failed.")
        }
        computeCommandEncoder.setBuffer(matrixBuffer, offset: 0, index: 4)
        
        //  Hardcoded to 32 for now (recommendation: read about threadExecutionWidth)
        let threadsPerGroup = MTLSize(width:32,height:1,depth:1)
        let numThreadgroups = MTLSize(width:((numNodes+1)+31)/32, height:1, depth:1)        //  Add one to guarantee the bias offset addition thread is performed
        computeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        
        computeCommandEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        //  Extract the updated weights
        let weightData = Data(bytesNoCopy: matrixBuffer.contents(), count: matrixByteLength, deallocator: .none)
        (weightData as NSData).getBytes(&weights, length:matrixByteLength)
    }
}

@available(OSX 10.11, iOS 8.0, *)
open class MetalNeuralNetwork {
    
    let device: MTLDevice?
    var commandQueue: MTLCommandQueue?
    var metalNNLibrary: MTLLibrary?
    
    //  Layers
    var layers : [MetalNeuralLayer]
    
    public init?(numInputs : Int, layerDefinitions: [(numNodes: Int, activation: NeuralActivationFunction)])
    {
        self.commandQueue = nil
        self.metalNNLibrary = nil
        self.layers = []
        
        // Get access to GPU
        self.device = MTLCreateSystemDefaultDevice()
        if device == nil { return nil }
        
        // Queue to handle an ordered list of command buffers
        self.commandQueue = device!.makeCommandQueue()
        
        // Access to Metal functions that are stored in MetalNeuralNetworkShaders string, e.g. sigmoid()
        self.metalNNLibrary = try? device!.makeLibrary(source: metalNNShaders, options: nil)
        if metalNNLibrary == nil { return nil }
        
        //  Set up the layers
        var numInputsFromPreviousLayer = numInputs
        for layerDefinition in layerDefinitions {
            let layer = MetalNeuralLayer(numInputs: numInputsFromPreviousLayer, layerDefinition: layerDefinition)
            layers.append(layer)
            numInputsFromPreviousLayer = layerDefinition.numNodes
        }
    }
    
    open func feedForward(_ inputs: [Float]) -> [Float] {
        //  Start with the inputs for the first layer
        var layerInputs = inputs
        
        //  Add a bias constant 1.0 to the input array
        layerInputs.append(1.0)
        
        //  Create a MTLBuffer for the input vector
        let inputsByteLength = layerInputs.count * MemoryLayout<Float>.size
        guard let device = device, var inputBuffer = device.makeBuffer(bytes: &layerInputs, length: inputsByteLength, options: MTLResourceOptions(rawValue: 0)) else {
            fatalError("inputBuffer: make buffer failed.∫")
        }
        
        //  Go through each layer
        for layer in layers {
            //  Calculate the outputs from the layer
            inputBuffer = layer.getLayerOutputs(inputBuffer, device: device, commandQueue: commandQueue!, metalLibrary: metalNNLibrary!)
        }
        
        return layers.last!.σ
    }
    
    
    open func trainOne(_ inputs: [Float], expectedOutputs: [Float], trainingRate: Float, weightDecay: Float)
    {
        //  Get the results of a feedForward run (each node remembers its own output)
        _ = feedForward(inputs)
        
        //  Calculate the delta for the final layer
        layers.last!.getFinalLayerDelta(expectedOutputs, device: device!, commandQueue: commandQueue!, metalLibrary: metalNNLibrary!)
        
        //  Get the deltas for the other layers
        if (layers.count > 1) {
            for nLayerIndex in stride(from: (layers.count - 2), through: 0, by: -1)
            {
                layers[nLayerIndex].getLayerDelta(layers[nLayerIndex+1], device: device!, commandQueue: commandQueue!, metalLibrary: metalNNLibrary!)
            }
        }
        
        //  Set the inputs for calculating the weight changes
        var layerInputs = inputs
        
        //  Add a bias constant 1.0 to the input array
        layerInputs.append(1.0)
        
        //  Create a MTLBuffer for the input vector
        var inputsByteLength = layerInputs.count * MemoryLayout<Float>.size
        
        guard let device = device, var inputBuffer = device.makeBuffer(bytes: &layerInputs, length: inputsByteLength, options: MTLResourceOptions(rawValue: 0)) else {
            fatalError("inputBuffer: make buffer failed.∫")
        }
        
        //  Go through each layer
        for nLayerIndex in 0..<layers.count {
            //  Update the weights in the layer
            layers[nLayerIndex].updateWeights(inputBuffer, trainingRate: trainingRate, weightDecay: weightDecay, device: device, commandQueue: commandQueue!, metalLibrary: metalNNLibrary!)
            
            if (nLayerIndex < layers.count - 1) {
                layerInputs = layers[nLayerIndex].σ
                layerInputs.append(1.0)
                
                inputsByteLength = layerInputs.count * MemoryLayout<Float>.size
                guard let newBuffer = device.makeBuffer(bytes: &layerInputs, length: inputsByteLength, options: MTLResourceOptions(rawValue: 0)) else {
                    fatalError("inputBuffer - newBuffer: make buffer failed.")
                }
                inputBuffer = newBuffer
            }
        }
    }

/*    public func testMetal()
    {
        
        //  Create a buffer for storing encoded commands that are sent to GPU
        let commandBuffer = commandQueue!.commandBuffer()
        
        //  Create an encoder for GPU commands
        let computeCommandEncoder = commandBuffer.computeCommandEncoder()
        
        //  Set up a compute pipeline with Sigmoid function and add it to encoder
        let sigmoidProgram = metalNNLibrary!.newFunctionWithName("sigmoid")
        var computePipelineFilter : MTLComputePipelineState?
        do {
            try computePipelineFilter = device!.newComputePipelineStateWithFunction(sigmoidProgram!)
        }
        catch {
            print("Error creating pipeline filter")
        }
        computeCommandEncoder.setComputePipelineState(computePipelineFilter!)
        
        var myvector = [Float](count: 123456, repeatedValue: 0)
        for (index, _) in myvector.enumerate() {
            myvector[index] = Float(index)
        }
        
        //  Calculate byte length of input data - myvector
        let myvectorByteLength = myvector.count*sizeofValue(myvector[0])
        
        //  Create a MTLBuffer - input data that the GPU and Metal and produce
        let inVectorBuffer = device!.newBufferWithBytes(&myvector, length: myvectorByteLength, options: MTLResourceOptions(rawValue: 0))
        
        //  Set the input vector for the Sigmoid() function, e.g. inVector
        //    atIndex: 0 here corresponds to buffer(0) in the Sigmoid function
        computeCommandEncoder.setBuffer(inVectorBuffer, offset: 0, atIndex: 0)
        
        //  Create the output vector for the Sigmoid() function, e.g. outVector
        //    atIndex: 1 here corresponds to buffer(1) in the Sigmoid function
        var resultdata = [Float](count:myvector.count, repeatedValue: 0)
        let outVectorBuffer = device!.newBufferWithBytes(&resultdata, length: myvectorByteLength, options: MTLResourceOptions(rawValue: 0))
        computeCommandEncoder.setBuffer(outVectorBuffer, offset: 0, atIndex: 1)
        
        //  Hardcoded to 32 for now (recommendation: read about threadExecutionWidth)
        let threadsPerGroup = MTLSize(width:32,height:1,depth:1)
        let numThreadgroups = MTLSize(width:(myvector.count+31)/32, height:1, depth:1)
        computeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        
        computeCommandEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        //  Get GPU data
        // outVectorBuffer.contents() returns UnsafeMutablePointer roughly equivalent to char* in C
        let data = NSData(bytesNoCopy: outVectorBuffer.contents(),
                    length: myvector.count*sizeof(Float), freeWhenDone: false)
        //  Prepare Swift array large enough to receive data from GPU
        var finalResultArray = [Float](count: myvector.count, repeatedValue: 0)
        
        //  Get data from GPU into Swift array
        data.getBytes(&finalResultArray, length:myvector.count * sizeof(Float))
    }
*/
}
