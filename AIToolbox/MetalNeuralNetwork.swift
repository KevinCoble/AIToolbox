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
        weights = [Float](count: numNodes * numWeights, repeatedValue: 0.0)
        
        //  Create the linear sum result array
        z = [Float](count: numNodes, repeatedValue: 0.0)
        
        //  Create the non-linear result array
        σ = [Float](count: numNodes, repeatedValue: 0.0)
        
        //  Create the delta array
        delta = [Float](count: numNodes, repeatedValue: 0.0)
        
        //  Initialize the weights
        for index in 0..<weights.count  {
            var standardDeviation = 1.0     //  Bias weights to one standard dev
            if ((index % numWeights) != numInputs) { standardDeviation = 1.0 / Double(numInputs) }
            weights[index] = MetalNeuralLayer.gaussianRandom(0.0, standardDeviation: standardDeviation)   //  input weights - Initialize to a random number to break initial symmetry of the network, scaled to the inputs
        }
    }
    
    static var y2 = 0.0
    static var use_last = false
    static func gaussianRandom(mean : Float, standardDeviation : Double) -> Float
    {
        var y1 : Double
        if (use_last)		        /* use value from previous call */
        {
            y1 = y2
            use_last = false
        }
        else
        {
            var w = 1.0
            var x1 = 0.0
            var x2 = 0.0
            repeat {
                x1 = 2.0 * (Double(arc4random()) / Double(UInt32.max)) - 1.0
                x2 = 2.0 * (Double(arc4random()) / Double(UInt32.max)) - 1.0
                w = x1 * x1 + x2 * x2
            } while ( w >= 1.0 )
            
            w = sqrt( (-2.0 * log( w ) ) / w )
            y1 = x1 * w
            y2 = x2 * w
            use_last = true
        }
        
        return mean + Float(y1 * standardDeviation)
    }
    
    
    func getLayerOutputs(inputs: MTLBuffer, device: MTLDevice, commandQueue: MTLCommandQueue, metalLibrary: MTLLibrary) -> MTLBuffer
    {
        //  Assume input array already has bias constant 1.0 appended
        //  Fully-connected nodes means all nodes get the same input array
        
        //  Create a buffer for storing encoded commands that are sent to GPU
        let commandBuffer = commandQueue.commandBuffer()
        
        //  Create an encoder for GPU commands
        let computeCommandEncoder = commandBuffer.computeCommandEncoder()
        
        //  Get the name of the metal shader for the layer's non-linearity
        var programName: String
        switch (activationFunction) {
        case .None:
            programName = "sumForward"
        case .HyperbolicTangent:
            programName = "tanhForward"
        case .Sigmoid:
            programName = "sigmoidForward"
        case .SigmoidWithCrossEntropy:
            programName = "sigmoidForward"
        case .RectifiedLinear:
            programName = "rectLinearForward"
        case .SoftSign:
            programName = "softSignForward"
        case .SoftMax:        //  Only valid on output (last) layer
            programName = "softMaxForward"
        }
        
        //  Set up a compute pipeline with activation function and add it to encoder
        let layerOutputProgram = metalLibrary.newFunctionWithName(programName)
        var computePipelineFilter : MTLComputePipelineState?
        do {
            try computePipelineFilter = device.newComputePipelineStateWithFunction(layerOutputProgram!)
        }
        catch {
            print("Error creating pipeline filter")
        }
        computeCommandEncoder.setComputePipelineState(computePipelineFilter!)
        
        //  Create a MTLBuffer for the weight matrix
        let matrixByteLength = weights.count * sizeofValue(weights[0])
        let matrixBuffer = device.newBufferWithBytes(&weights, length: matrixByteLength, options: MTLResourceOptions(rawValue: 0))
        
        //  Set the input vector for the activation function, e.g. inputs
        //    atIndex: 0 here corresponds to buffer(0) in the activation function
        computeCommandEncoder.setBuffer(inputs, offset: 0, atIndex: 0)
        
        //  Set the matrix vector for the activation function, e.g. matrix
        //    atIndex: 1 here corresponds to buffer(1) in the activation function
        computeCommandEncoder.setBuffer(matrixBuffer, offset: 0, atIndex: 1)
        
        //  Create the output vector for the summation function, e.g. zBuffer
        //    atIndex: 2 here corresponds to buffer(2) in the activation function
        let sumResultByteLength = z.count * sizeof(Float)
        let zBuffer = device.newBufferWithLength(sumResultByteLength, options: MTLResourceOptions(rawValue: 0))
        computeCommandEncoder.setBuffer(zBuffer, offset: 0, atIndex: 2)
        
        //  Create the output vector for the activation function, e.g. σBuffer
        //    atIndex: 3 here corresponds to buffer(3) in the activation function
        let nextLayerInputByteLength = (σ.count + 1) * sizeof(Float)       //  Add one for the bias term
        let σBuffer = device.newBufferWithLength(nextLayerInputByteLength, options: MTLResourceOptions(rawValue: 0))
        computeCommandEncoder.setBuffer(σBuffer, offset: 0, atIndex: 3)
        
        //  Create the sizing array
        let sizeArray : [Int32] = [Int32(numNodes), Int32(numWeights)]
        let sizeArrayByteLength = sizeArray.count * sizeof(Int32)
        let sizeBuffer = device.newBufferWithBytes(sizeArray, length: sizeArrayByteLength, options: MTLResourceOptions(rawValue: 0))
        computeCommandEncoder.setBuffer(sizeBuffer, offset: 0, atIndex: 4)
        
        //  Hardcoded to 32 for now (recommendation: read about threadExecutionWidth)
        let threadsPerGroup = MTLSize(width:32,height:1,depth:1)
        let numThreadgroups = MTLSize(width:((numNodes+1)+31)/32, height:1, depth:1)        //  Add one to guarantee the bias offset addition thread is performed
        computeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        
        computeCommandEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        //  Extract the z values
        let zData = NSData(bytesNoCopy: zBuffer.contents(), length: sumResultByteLength, freeWhenDone: false)
        zData.getBytes(&z, length:sumResultByteLength)
        
        //  Extract the σ values
        let activationResultByteLength = σ.count * sizeof(Float)
        let σData = NSMutableData(bytesNoCopy: σBuffer.contents(), length: activationResultByteLength, freeWhenDone: false)
        σData.getBytes(&σ, length:activationResultByteLength)
        
        //  Add the bias term to the buffer so it is ready for the next layer
        let byteRange = NSMakeRange(activationResultByteLength, sizeof(Float))
        var bias : Float = 1.0
        σData.replaceBytesInRange(byteRange, withBytes: &bias)
        
        
        if (activationFunction == .SoftMax) {
            //  Get the sum of the output nodes
            var sum: Float = 0.0;
            vDSP_vswsum(σ, 1, &sum, 1, 1, vDSP_Length(numNodes));
            
            //  Divide each element by the sum
            vDSP_vsdiv(σ, 1, &sum, &σ, 1, vDSP_Length(numNodes));
            
            //  Get the outputs into the passed-back buffer
            let byteRange = NSMakeRange(0, activationResultByteLength)
            σData.replaceBytesInRange(byteRange, withBytes: &σ)
        }
        
        return σBuffer
    }
    
    //  Get the partial derivitive of the error with respect to the weighted sum
    func getFinalLayerDelta(expectedOutputs: [Float], device: MTLDevice, commandQueue: MTLCommandQueue, metalLibrary: MTLLibrary)
    {
        //  error = (result - expected value)^2  (squared error) - not the case for softmax or cross entropy
        //  derivitive of error = 2 * (result - expected value) * result'  (chain rule - result is a function of the sum through the non-linearity)
        //  derivitive of the non-linearity: tanh' -> 1 - result^2, sigmoid -> result - result^2, rectlinear -> 0 if result<0 else 1
        //  derivitive of error = 2 * (result - expected value) * derivitive from above

        //  Create a buffer for storing encoded commands that are sent to GPU
        let commandBuffer = commandQueue.commandBuffer()
        
        //  Create an encoder for GPU commands
        let computeCommandEncoder = commandBuffer.computeCommandEncoder()
        
        //  Get the name of the metal shader for the layer's non-linearity
        var programName: String
        switch (activationFunction) {
        case .None:
            programName = "sumFinal"
        case .HyperbolicTangent:
            programName = "tanhFinal"
        case .Sigmoid:
            programName = "sigmoidFinal"
        case .SigmoidWithCrossEntropy:
            programName = "sigmoidCrossEntropyFinal"
        case .RectifiedLinear:
            programName = "rectLinearFinal"
        case .SoftSign:
            programName = "softSignFinal"
        case .SoftMax:        //  Only valid on output (last) layer
            programName = "softMaxFinal"
        }
        
        //  Set up a compute pipeline with activation function and add it to encoder
        let layerOutputProgram = metalLibrary.newFunctionWithName(programName)
        var computePipelineFilter : MTLComputePipelineState?
        do {
            try computePipelineFilter = device.newComputePipelineStateWithFunction(layerOutputProgram!)
        }
        catch {
            print("Error creating pipeline filter")
        }
        computeCommandEncoder.setComputePipelineState(computePipelineFilter!)
        
        //  Create a MTLBuffer for the σ vector
        let resultsByteLength = σ.count * sizeof(Float)
        let σBuffer = device.newBufferWithBytes(&σ, length: resultsByteLength, options: MTLResourceOptions(rawValue: 0))
        
        //  Set the result vector for the final delta function, e.g. σBuffer
        //    atIndex: 0 here corresponds to buffer(0) in the activation function
        computeCommandEncoder.setBuffer(σBuffer, offset: 0, atIndex: 0)
        
        //  Create a MTLBuffer for the expected results vector
        let expectedBuffer = device.newBufferWithBytes(expectedOutputs, length: resultsByteLength, options: MTLResourceOptions(rawValue: 0))
        
        //  Set the result vector for the sigma value, e.g. expectedBuffer
        //    atIndex: 0 here corresponds to buffer(0) in the activation function
        computeCommandEncoder.setBuffer(expectedBuffer, offset: 0, atIndex: 1)
        
        //  Create the output vector for the final delta value, e.g. deltaBuffer
        //    atIndex: 3 here corresponds to buffer(3) in the activation function
        let deltaBuffer = device.newBufferWithLength(resultsByteLength, options: MTLResourceOptions(rawValue: 0))
        computeCommandEncoder.setBuffer(deltaBuffer, offset: 0, atIndex: 2)
        
        //  Hardcoded to 32 for now (recommendation: read about threadExecutionWidth)
        let threadsPerGroup = MTLSize(width:32,height:1,depth:1)
        let numThreadgroups = MTLSize(width:((numNodes+1)+31)/32, height:1, depth:1)        //  Add one to guarantee the bias offset addition thread is performed
        computeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        
        computeCommandEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        //  Extract the delta values
        let deltaData = NSData(bytesNoCopy: deltaBuffer.contents(), length: resultsByteLength, freeWhenDone: false)
        deltaData.getBytes(&delta, length:resultsByteLength)
    }
    
    //  Transfer the delta from the forward layer to this one
    func getLayerDelta(nextLayer: MetalNeuralLayer, device: MTLDevice, commandQueue: MTLCommandQueue, metalLibrary: MTLLibrary)
    {
        //  Create a buffer for storing encoded commands that are sent to GPU
        let commandBuffer = commandQueue.commandBuffer()
        
        //  Create an encoder for GPU commands
        let computeCommandEncoder = commandBuffer.computeCommandEncoder()
        
        //  Get the name of the metal shader for the layer's delta calculation
        var programName: String
        switch (activationFunction) {
        case .None:
            programName = "sumDelta"
        case .HyperbolicTangent:
            programName = "tanhDelta"
        case .Sigmoid:
            programName = "sigmoidDelta"
        case .SigmoidWithCrossEntropy:
            programName = "sigmoidDelta"
        case .RectifiedLinear:
            programName = "rectLinearDelta"
        case .SoftSign:
            programName = "softSignDelta"
        case .SoftMax:        //  Only valid on output (last) layer
            print("SoftMax activation function on non-last layer")
            return
        }
        
        //  Set up a compute pipeline with activation function and add it to encoder
        let layerOutputProgram = metalLibrary.newFunctionWithName(programName)
        var computePipelineFilter : MTLComputePipelineState?
        do {
            try computePipelineFilter = device.newComputePipelineStateWithFunction(layerOutputProgram!)
        }
        catch {
            print("Error creating pipeline filter")
        }
        computeCommandEncoder.setComputePipelineState(computePipelineFilter!)
        
        //  Create a MTLBuffer for the weight matrix from the next layer and assign to buffer 0
        let weightsByteLength = nextLayer.weights.count * sizeof(Float)
        let weightBuffer = device.newBufferWithBytes(&nextLayer.weights, length: weightsByteLength, options: MTLResourceOptions(rawValue: 0))
        computeCommandEncoder.setBuffer(weightBuffer, offset: 0, atIndex: 0)
        
        //  Create a MTLBuffer for the delta vector from the next layer and assign to buffer 1
        let deltaByteLength = nextLayer.delta.count * sizeof(Float)
        let deltaBuffer = device.newBufferWithBytes(&nextLayer.delta, length: deltaByteLength, options: MTLResourceOptions(rawValue: 0))
        computeCommandEncoder.setBuffer(deltaBuffer, offset: 0, atIndex: 1)
        
        //  Create a MTLBuffer for the sizing array from the next layer and assign to buffer 2
        let sizeArray : [Int32] = [Int32(nextLayer.numNodes), Int32(numNodes)]
        let sizeArrayByteLength = sizeArray.count * sizeof(Int32)
        let sizeBuffer = device.newBufferWithBytes(sizeArray, length: sizeArrayByteLength, options: MTLResourceOptions(rawValue: 0))
        computeCommandEncoder.setBuffer(sizeBuffer, offset: 0, atIndex: 2)
        
        //  Create a MTLBuffer for the σ vector and assign to buffer 3
        let resultsByteLength = delta.count * sizeof(Float)
        let σBuffer = device.newBufferWithBytes(&σ, length: resultsByteLength, options: MTLResourceOptions(rawValue: 0))
        computeCommandEncoder.setBuffer(σBuffer, offset: 0, atIndex: 3)
        
        //  Create the output vector for the final delta function, e.g. resultBuffer and assignt to buffer 4
        //    atIndex: 3 here corresponds to buffer(3) in the activation function
        let resultBuffer = device.newBufferWithLength(resultsByteLength, options: MTLResourceOptions(rawValue: 0))
        computeCommandEncoder.setBuffer(resultBuffer, offset: 0, atIndex: 4)
        
        //  Hardcoded to 32 for now (recommendation: read about threadExecutionWidth)
        let threadsPerGroup = MTLSize(width:32,height:1,depth:1)
        let numThreadgroups = MTLSize(width:((numNodes+1)+31)/32, height:1, depth:1)        //  Add one to guarantee the bias offset addition thread is performed
        computeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        
        computeCommandEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        //  Extract the delta values
        let deltaData = NSData(bytesNoCopy: deltaBuffer.contents(), length: resultsByteLength, freeWhenDone: false)
        deltaData.getBytes(&delta, length:resultsByteLength)
    }
    
    func updateWeights(inputs: MTLBuffer, trainingRate: Float, weightDecay: Float, device: MTLDevice, commandQueue: MTLCommandQueue, metalLibrary: MTLLibrary)
    {
        //  Assume input array already has bias constant 1.0 appended
        //  Fully-connected nodes means all nodes get the same input array
        //  Create a buffer for storing encoded commands that are sent to GPU
        let commandBuffer = commandQueue.commandBuffer()
        
        //  Create an encoder for GPU commands
        let computeCommandEncoder = commandBuffer.computeCommandEncoder()
        
        //  Get the name of the metal shader for the layer's weight update
        let programName = "updateWeights"
        
        //  Set up a compute pipeline with activation function and add it to encoder
        let layerOutputProgram = metalLibrary.newFunctionWithName(programName)
        var computePipelineFilter : MTLComputePipelineState?
        do {
            try computePipelineFilter = device.newComputePipelineStateWithFunction(layerOutputProgram!)
        }
        catch {
            print("Error creating pipeline filter")
        }
        computeCommandEncoder.setComputePipelineState(computePipelineFilter!)
        
        //  Set the input vector as the first buffer
        computeCommandEncoder.setBuffer(inputs, offset: 0, atIndex: 0)
        
        //  Create a MTLBuffer for the delta vector and assign to buffer 1
        let deltaByteLength = delta.count * sizeof(Float)
        let deltaBuffer = device.newBufferWithBytes(&delta, length: deltaByteLength, options: MTLResourceOptions(rawValue: 0))
        computeCommandEncoder.setBuffer(deltaBuffer, offset: 0, atIndex: 1)
        
        //  Create the parameter array
        let paramArray : [Float] = [trainingRate, weightDecay]
        let paramArrayByteLength = paramArray.count * sizeof(Float)
        let paramBuffer = device.newBufferWithBytes(paramArray, length: paramArrayByteLength, options: MTLResourceOptions(rawValue: 0))
        computeCommandEncoder.setBuffer(paramBuffer, offset: 0, atIndex: 2)
        
        //  Create the sizing array
        let sizeArray : [Int32] = [Int32(numNodes), Int32(numWeights)]
        let sizeArrayByteLength = sizeArray.count * sizeof(Int32)
        let sizeBuffer = device.newBufferWithBytes(sizeArray, length: sizeArrayByteLength, options: MTLResourceOptions(rawValue: 0))
        computeCommandEncoder.setBuffer(sizeBuffer, offset: 0, atIndex: 3)
        
        //  Create a MTLBuffer for the weight matrix
        let matrixByteLength = weights.count * sizeofValue(weights[0])
        let matrixBuffer = device.newBufferWithBytes(&weights, length: matrixByteLength, options: MTLResourceOptions(rawValue: 0))
        computeCommandEncoder.setBuffer(matrixBuffer, offset: 0, atIndex: 4)
        
        //  Hardcoded to 32 for now (recommendation: read about threadExecutionWidth)
        let threadsPerGroup = MTLSize(width:32,height:1,depth:1)
        let numThreadgroups = MTLSize(width:((numNodes+1)+31)/32, height:1, depth:1)        //  Add one to guarantee the bias offset addition thread is performed
        computeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        
        computeCommandEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        //  Extract the updated weights
        let weightData = NSData(bytesNoCopy: matrixBuffer.contents(), length: matrixByteLength, freeWhenDone: false)
        weightData.getBytes(&weights, length:matrixByteLength)
    }
}

@available(OSX 10.11, iOS 8.0, *)
public class MetalNeuralNetwork {
    
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
        self.commandQueue = device!.newCommandQueue()
        
        // Access to Metal functions that are stored in MetalNeuralNetworkShaders string, e.g. sigmoid()
        self.metalNNLibrary = try? device!.newLibraryWithSource(metalNNShaders, options: nil)
        if metalNNLibrary == nil { return nil }
        
        //  Set up the layers
        var numInputsFromPreviousLayer = numInputs
        for layerDefinition in layerDefinitions {
            let layer = MetalNeuralLayer(numInputs: numInputsFromPreviousLayer, layerDefinition: layerDefinition)
            layers.append(layer)
            numInputsFromPreviousLayer = layerDefinition.numNodes
        }
    }
    
    public func feedForward(inputs: [Float]) -> [Float] {
        //  Start with the inputs for the first layer
        var layerInputs = inputs
        
        //  Add a bias constant 1.0 to the input array
        layerInputs.append(1.0)
        
        //  Create a MTLBuffer for the input vector
        let inputsByteLength = layerInputs.count * sizeof(Float)
        var inputBuffer = device!.newBufferWithBytes(&layerInputs, length: inputsByteLength, options: MTLResourceOptions(rawValue: 0))
        
        //  Go through each layer
        for layer in layers {
            //  Calculate the outputs from the layer
            inputBuffer = layer.getLayerOutputs(inputBuffer, device: device!, commandQueue: commandQueue!, metalLibrary: metalNNLibrary!)
        }
        
        return layers.last!.σ
    }
    
    
    public func trainOne(inputs: [Float], expectedOutputs: [Float], trainingRate: Float, weightDecay: Float)
    {
        //  Get the results of a feedForward run (each node remembers its own output)
        feedForward(inputs)
        
        //  Calculate the delta for the final layer
        layers.last!.getFinalLayerDelta(expectedOutputs, device: device!, commandQueue: commandQueue!, metalLibrary: metalNNLibrary!)
        
        //  Get the deltas for the other layers
        if (layers.count > 1) {
            for nLayerIndex in (layers.count - 2).stride(through: 0, by: -1)
            {
                layers[nLayerIndex].getLayerDelta(layers[nLayerIndex+1], device: device!, commandQueue: commandQueue!, metalLibrary: metalNNLibrary!)
            }
        }
        
        //  Set the inputs for calculating the weight changes
        var layerInputs = inputs
        
        //  Add a bias constant 1.0 to the input array
        layerInputs.append(1.0)
        
        //  Create a MTLBuffer for the input vector
        var inputsByteLength = layerInputs.count * sizeof(Float)
        var inputBuffer = device!.newBufferWithBytes(&layerInputs, length: inputsByteLength, options: MTLResourceOptions(rawValue: 0))
        
        //  Go through each layer
        for nLayerIndex in 0..<layers.count {
            //  Update the weights in the layer
            layers[nLayerIndex].updateWeights(inputBuffer, trainingRate: trainingRate, weightDecay: weightDecay, device: device!, commandQueue: commandQueue!, metalLibrary: metalNNLibrary!)
            
            if (nLayerIndex < layers.count - 1) {
                layerInputs = layers[nLayerIndex].σ
                layerInputs.append(1.0)
                
                inputsByteLength = layerInputs.count * sizeof(Float)
                inputBuffer = device!.newBufferWithBytes(&layerInputs, length: inputsByteLength, options: MTLResourceOptions(rawValue: 0))
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
