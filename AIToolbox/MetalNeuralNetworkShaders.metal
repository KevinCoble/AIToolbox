//
//  MetalNeuralNetworkShaders.metal
//  AIToolbox
//
//  Created by Kevin Coble on 1/7/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

//  This is duplicated as a string in the .swift version of the file
//  This file is left as a way to test the compile of the shaders at build time

#include <metal_stdlib>
using namespace metal;


kernel void sumForward(const device float *inMatrix [[ buffer(0) ]],
                const device float *inInputs [[ buffer(1)]],
                    device float *sumOutVector [[ buffer(2) ]],
                    device float *activationOutVector [[ buffer(3) ]],
                    device int *sizingArray [[ buffer(4) ]],
                    uint id [[ thread_position_in_grid ]])
{
    int index;
    float sum = 0.0;
    
    //  Get the number of weights for each node
    int numWeights = sizingArray[1];
    
    //  Get the weight offset for this node
    int weightOffset = numWeights * id;
    
    //  Fet the weighted sum
    for (index = 0; index < numWeights; index++) {
        sum += inMatrix[weightOffset] * inInputs[index];
        weightOffset++;
    }
    sumOutVector[id] = sum;
    
    //  No activation function, copy sum (z) to activation (σ)
    activationOutVector[id] = sum;
}

kernel void sigmoidForward(const device float *inMatrix [[ buffer(0) ]],
                    const device float *inInputs [[ buffer(1)]],
                    device float *sumOutVector [[ buffer(2) ]],
                    device float *activationOutVector [[ buffer(3) ]],
                    device int *sizingArray [[ buffer(4) ]],
                    uint id [[ thread_position_in_grid ]])
{
    int index;
    float sum = 0.0;
    
    //  Get the number of weights for each node
    int numWeights = sizingArray[1];
    
    //  Get the weight offset for this node
    int weightOffset = numWeights * id;
    
    //  Fet the weighted sum
    for (index = 0; index < numWeights; index++) {
        sum += inMatrix[weightOffset] * inInputs[index];
        weightOffset++;
    }
    sumOutVector[id] = sum;
    
    // This calculates sigmoid for the node
    activationOutVector[id] = 1.0 / (1.0 + exp(-sum));
}

kernel void tanhForward(const device float *inMatrix [[ buffer(0) ]],
                    const device float *inInputs [[ buffer(1)]],
                    device float *sumOutVector [[ buffer(2) ]],
                    device float *activationOutVector [[ buffer(3) ]],
                    device int *sizingArray [[ buffer(4) ]],
                    uint id [[ thread_position_in_grid ]])
{
    int index;
    float sum = 0.0;
    
    //  Get the number of weights for each node
    int numWeights = sizingArray[1];
    
    //  Get the weight offset for this node
    int weightOffset = numWeights * id;
    
    //  Fet the weighted sum
    for (index = 0; index < numWeights; index++) {
        sum += inMatrix[weightOffset] * inInputs[index];
        weightOffset++;
    }
    sumOutVector[id] = sum;
    
    // This calculates hyperbolic tangent for the node
    activationOutVector[id] = tanh(sum);
}

kernel void rectLinearForward(const device float *inMatrix [[ buffer(0) ]],
                        const device float *inInputs [[ buffer(1)]],
                        device float *sumOutVector [[ buffer(2) ]],
                        device float *activationOutVector [[ buffer(3) ]],
                        device int *sizingArray [[ buffer(4) ]],
                        uint id [[ thread_position_in_grid ]])
{
    int index;
    float sum = 0.0;
    
    //  Get the number of weights for each node
    int numWeights = sizingArray[1];
    
    //  Get the weight offset for this node
    int weightOffset = numWeights * id;
    
    //  Fet the weighted sum
    for (index = 0; index < numWeights; index++) {
        sum += inMatrix[weightOffset] * inInputs[index];
        weightOffset++;
    }
    sumOutVector[id] = sum;
    
    // This calculates rectified linear value for the node
    activationOutVector[id] = sum;
    if (sum < 0.0) activationOutVector[id] = 0.0;
}

kernel void softSignForward(const device float *inMatrix [[ buffer(0) ]],
                              const device float *inInputs [[ buffer(1)]],
                              device float *sumOutVector [[ buffer(2) ]],
                              device float *activationOutVector [[ buffer(3) ]],
                              device int *sizingArray [[ buffer(4) ]],
                              uint id [[ thread_position_in_grid ]])
{
    int index;
    float sum = 0.0;
    
    //  Get the number of weights for each node
    int numWeights = sizingArray[1];
    
    //  Get the weight offset for this node
    int weightOffset = numWeights * id;
    
    //  Fet the weighted sum
    for (index = 0; index < numWeights; index++) {
        sum += inMatrix[weightOffset] * inInputs[index];
        weightOffset++;
    }
    sumOutVector[id] = sum;
    
    // This calculates soft sign for the node
    activationOutVector[id] = sum / (1.0 + abs(sum));
}

kernel void softMaxForward(const device float *inMatrix [[ buffer(0) ]],
                            const device float *inInputs [[ buffer(1)]],
                            device float *sumOutVector [[ buffer(2) ]],
                            device float *activationOutVector [[ buffer(3) ]],
                            device int *sizingArray [[ buffer(4) ]],
                            uint id [[ thread_position_in_grid ]])
{
    int index;
    float sum = 0.0;
    
    //  Get the number of weights for each node
    int numWeights = sizingArray[1];
    
    //  Get the weight offset for this node
    int weightOffset = numWeights * id;
    
    //  Get the weighted sum
    for (index = 0; index < numWeights; index++) {
        sum += inMatrix[weightOffset] * inInputs[index];
        weightOffset++;
    }
    sumOutVector[id] = sum;
    
    // This calculates the exponent for the node (must be summed across all nodes later
    activationOutVector[id] = exp(sum);
}

kernel void sumFinal(const device float *inSigma [[ buffer(0) ]],
                           const device float *inExpected [[ buffer(1)]],
                           device float *outDelta [[ buffer(2) ]],
                           uint id [[ thread_position_in_grid ]])
{
    outDelta[id] = 2.0 * (inSigma[id] - inExpected[id]);
}

kernel void tanhFinal(const device float *inSigma [[ buffer(0) ]],
                     const device float *inExpected [[ buffer(1)]],
                     device float *outDelta [[ buffer(2) ]],
                     uint id [[ thread_position_in_grid ]])
{
    outDelta[id] = 2.0 * (inSigma[id] - inExpected[id]) * (1.0 - inSigma[id] * inSigma[id]);
}

kernel void sigmoidFinal(const device float *inSigma [[ buffer(0) ]],
                      const device float *inExpected [[ buffer(1)]],
                      device float *outDelta [[ buffer(2) ]],
                      uint id [[ thread_position_in_grid ]])
{
    outDelta[id] = 2.0 * (inSigma[id] - inExpected[id]) * (inSigma[id] - inSigma[id] * inSigma[id]);
}

kernel void sigmoidCrossEntropyFinal(const device float *inSigma [[ buffer(0) ]],
                     const device float *inExpected [[ buffer(1)]],
                     device float *outDelta [[ buffer(2) ]],
                     uint id [[ thread_position_in_grid ]])
{
    outDelta[id] = (inSigma[id] - inExpected[id]);
}

kernel void rectLinearFinal(const device float *inSigma [[ buffer(0) ]],
                                     const device float *inExpected [[ buffer(1)]],
                                     device float *outDelta [[ buffer(2) ]],
                                     uint id [[ thread_position_in_grid ]])
{
    if (inSigma[id] < 0.0) {
        outDelta[id] = 0.0;
    }
    else {
        outDelta[id] = 2.0 * (inSigma[id] - inExpected[id]);
    }
}

kernel void softSignFinal(const device float *inSigma [[ buffer(0) ]],
                                     const device float *inExpected [[ buffer(1)]],
                                     device float *outDelta [[ buffer(2) ]],
                                     uint id [[ thread_position_in_grid ]])
{
    float result;
    result = (1.0 - abs(inSigma[id]));
    result *= result;
    result *= 2.0 * (inSigma[id] - inExpected[id]);
    outDelta[id] = result;
}

kernel void softMaxFinal(const device float *inSigma [[ buffer(0) ]],
                                     const device float *inExpected [[ buffer(1)]],
                                     device float *outDelta [[ buffer(2) ]],
                                     uint id [[ thread_position_in_grid ]])
{
    outDelta[id] = (inSigma[id] - inExpected[id]);
}

void layerDelta(uint id, int numNodesThisLayer, int numNodesNextLayer,
                const device float *nextWeights, const device float *nextDelta, device float *outDelta);

kernel void sumDelta(const device float *nextLayerWeights [[ buffer(0) ]],
                     const device float *nextLayerDelta [[ buffer(1) ]],
                     const device int *sizingArray [[ buffer(2) ]],
                     const device float *inSigma [[ buffer(3) ]],
                     device float *outDelta [[ buffer(4) ]],
                     uint id [[ thread_position_in_grid ]])
{
    //  Extract the sizes from the sizing array
    int numNodesNextLayer = sizingArray[0];
    int numNodesThisLayer = sizingArray[1];
    
    //  Calculate the delta for this layer
    layerDelta(id, numNodesThisLayer, numNodesNextLayer, nextLayerWeights, nextLayerDelta, outDelta);
    
    //  Multiply the delta by the non-linearity - but there is none for sum
}

kernel void tanhDelta(const device float *nextLayerWeights [[ buffer(0) ]],
                     const device float *nextLayerDelta [[ buffer(1) ]],
                     const device int *sizingArray [[ buffer(2) ]],
                     const device float *inSigma [[ buffer(3) ]],
                     device float *outDelta [[ buffer(4) ]],
                     uint id [[ thread_position_in_grid ]])
{
    //  Extract the sizes from the sizing array
    int numNodesNextLayer = sizingArray[0];
    int numNodesThisLayer = sizingArray[1];
    
    //  Calculate the delta for this layer
    layerDelta(id, numNodesThisLayer, numNodesNextLayer, nextLayerWeights, nextLayerDelta, outDelta);
    
    //  Multiply the delta by the non-linearity
    outDelta[id] *= (1 - inSigma[id] * inSigma[id]);
}

kernel void sigmoidDelta(const device float *nextLayerWeights [[ buffer(0) ]],
                      const device float *nextLayerDelta [[ buffer(1) ]],
                      const device int *sizingArray [[ buffer(2) ]],
                      const device float *inSigma [[ buffer(3) ]],
                      device float *outDelta [[ buffer(4) ]],
                      uint id [[ thread_position_in_grid ]])
{
    //  Extract the sizes from the sizing array
    int numNodesNextLayer = sizingArray[0];
    int numNodesThisLayer = sizingArray[1];
    
    //  Calculate the delta for this layer
    layerDelta(id, numNodesThisLayer, numNodesNextLayer, nextLayerWeights, nextLayerDelta, outDelta);
    
    //  Multiply the delta by the non-linearity
    outDelta[id] *= (inSigma[id] - inSigma[id] * inSigma[id]);
}

kernel void rectLinearDelta(const device float *nextLayerWeights [[ buffer(0) ]],
                         const device float *nextLayerDelta [[ buffer(1) ]],
                         const device int *sizingArray [[ buffer(2) ]],
                         const device float *inSigma [[ buffer(3) ]],
                         device float *outDelta [[ buffer(4) ]],
                         uint id [[ thread_position_in_grid ]])
{
    //  Extract the sizes from the sizing array
    int numNodesNextLayer = sizingArray[0];
    int numNodesThisLayer = sizingArray[1];
    
    //  Calculate the delta for this layer
    layerDelta(id, numNodesThisLayer, numNodesNextLayer, nextLayerWeights, nextLayerDelta, outDelta);
    
    //  Multiply the delta by the non-linearity
    outDelta[id] = inSigma[id] < 0.0 ? 0.0 : outDelta[id];
}

kernel void softSignDelta(const device float *nextLayerWeights [[ buffer(0) ]],
                            const device float *nextLayerDelta [[ buffer(1) ]],
                            const device int *sizingArray [[ buffer(2) ]],
                            const device float *inSigma [[ buffer(3) ]],
                            device float *outDelta [[ buffer(4) ]],
                            uint id [[ thread_position_in_grid ]])
{
    //  Extract the sizes from the sizing array
    int numNodesNextLayer = sizingArray[0];
    int numNodesThisLayer = sizingArray[1];
    
    //  Calculate the delta for this layer
    layerDelta(id, numNodesThisLayer, numNodesNextLayer, nextLayerWeights, nextLayerDelta, outDelta);
    
    //  Multiply the delta by the non-linearity
    if (inSigma[id] < 0) outDelta[id] *= -1.0;
    outDelta[id] /= (1.0 + inSigma[id]) * (1.0 + inSigma[id]);
}

void layerDelta(uint thisLayerNode, int numNodesThisLayer, int numNodesNextLayer,
                const device float *nextWeights, const device float *nextDelta, device float *outDelta)
{
    int nextLayerNode;
    int weightOffset;

    //  Reset delta
    outDelta[thisLayerNode] = 0.0;
        
    //  Add each portion from the nodes in the next forward layer
    for (nextLayerNode = 0; nextLayerNode < numNodesNextLayer; nextLayerNode++) {
        weightOffset = (numNodesThisLayer + 1) * nextLayerNode + thisLayerNode;
        outDelta[thisLayerNode] += nextWeights[weightOffset] * nextDelta[nextLayerNode];
    }
}

kernel void updateWeights(const device float *inputs [[ buffer(0) ]],
                   const device float *delta [[ buffer(1) ]],
                   const device float *parameters [[ buffer(2) ]],
                   const device int *sizingArray [[ buffer(3) ]],
                   device float *weights [[ buffer(4) ]],
                   uint id [[ thread_position_in_grid ]])
{
    int index;
    
    //  Get the number of weights for each node
    int numWeights = sizingArray[1];
    
    //  Get the weight offset for this node
    int weightOffset = numWeights * id;
    
    float weightDecay = parameters[1];
    if (weightDecay < 1.0) {
        for (index = 0; index < numWeights; index++) {
            weights[weightOffset+index] *= weightDecay;
        }
    }
    //  weights = weights + delta * inputs * training rate
    float trainingWeight = parameters[0];
    for (index = 0; index < numWeights; index++) {
        weights[weightOffset] -= delta[id] * inputs[index] * trainingWeight;
        weightOffset++;
    }
}

