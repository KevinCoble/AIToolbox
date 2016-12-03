/*: Introduction
 # Neural Networks
 
 This playground shows the basics of Feed-Forward Neural Networks, and shows how to use the AIToolbox framework to generate training data, train a neural network, and plot the results
 
 We start with the required import statements.  If this was your own program, rather than a playground, you add an 'import AIToolbox' line instead of the 'import PlaygroundSupport' line.
 */
import UIKit
import PlaygroundSupport

/*: Examples
 This playground shows multiple examples of neural networks in action.  The following integer index switches between the examples.  Start at 1, and change it to higher numbers as the narrative requests
 */
let exampleIndex = 1

/*: Data Set
 ## Create a Classification DataSet
 
 We need a data set to train our Neural Network on
 
 DataSets in AIToolbox come in two types, Regression and Classification.  Regression is where you get a value (or array of values if the output dimension is >1) for your inputs.  Classification is when you get an integer 'class label' based on the inputs.  Regression is used to do things like 'predict time till failure', while classification is used for tasks like 'is that tumor malignant or benign'
 
 This is a Neural Network, which can be either a regressor or classifier.  We will be doing classification for this example, so our data set will be the 'Classification' type, with an input input vector sized for two variables (the X and Y axis on our plot later).
 */
let data = DataSet(dataType: .classification, inputDimension: 2, outputDimension: 1)
/*:
 Add a few data points to the data set.  Note the 'output' category is a single integer.  This is the 'class label' for the point.  All points with the same label are assumed to be in the same class.
 We will start with linearly seperable data for the initial example
 */
try data.addDataPoint(input: [0.3, 0.9], dataClass:0)
try data.addDataPoint(input: [0.8, 0.3], dataClass:0)
try data.addDataPoint(input: [0.5, 0.6], dataClass:0)
try data.addDataPoint(input: [0.2, 0.7], dataClass:1)
try data.addDataPoint(input: [0.2, 0.3], dataClass:1)
try data.addDataPoint(input: [0.4, 0.5], dataClass:1)

//  Additional points for later examples
if (exampleIndex > 1) {
    try data.addDataPoint(input: [0.3, 0.1], dataClass:0)
}
if (exampleIndex > 3) {
    try data.addDataPoint(input: [0.8, 0.8], dataClass:1)
    try data.addDataPoint(input: [0.9, 0.8], dataClass:1)
    try data.addDataPoint(input: [0.8, 0.9], dataClass:1)
}

/*: Neural Network
 ## Create a Neural Network
 
 We now create a network.  A network is created as a series of 'layers'.  Each layer is a set of 'nodes'.  The nodes take the data from the previous layer (or the direct inputs if it is the first layer), processes the information based on 'learned' weights, and sends the data to the next layer.  The data from final layer (called the 'output' layer') can be processed to be a value for regression, or a label for classification.
 The number of nodes and the number of layers affect what can be learned.  To start with, we will begin with a single node in a single layer to show these limitations.
 The processing a node does usually needs to be non-linear, else the network can have problems learning complicated functions.  The non-linearity part is called the activation function.  This function is often one that limits the output to a fixed range, even if the input values are very large.  Therefore, the sigmoid function or hyperbolic tangent are usually used.  We will use the sigmoid function for this example.
 */
var network : NeuralNetwork
if (exampleIndex <= 2) {
    //  One node in one layer
    network = NeuralNetwork(numInputs: 2, layerDefinitions: [(layerType: .simpleFeedForward, numNodes: 1, activation: NeuralActivationFunction.sigmoid, auxiliaryData: nil)])
}
    
    //  Network creation for later examples
else if (exampleIndex == 3 || exampleIndex == 4) {
    //  Two nodes in first layer, one in second layer
    network = NeuralNetwork(numInputs: 2, layerDefinitions: [(layerType: .simpleFeedForward, numNodes: 2, activation: NeuralActivationFunction.sigmoid, auxiliaryData: nil),
                                                             (layerType: .simpleFeedForward, numNodes: 1, activation:NeuralActivationFunction.sigmoid, auxiliaryData: nil)])
}
else {
    //  8 nodes in first layer, three in second, and one in last layer
    network = NeuralNetwork(numInputs: 2, layerDefinitions: [(layerType: .simpleFeedForward, numNodes: 8, activation: NeuralActivationFunction.sigmoid, auxiliaryData: nil),
                                                             (layerType: .simpleFeedForward, numNodes: 3, activation: NeuralActivationFunction.sigmoid, auxiliaryData: nil),
                                                             (layerType: .simpleFeedForward, numNodes: 1, activation:NeuralActivationFunction.sigmoid, auxiliaryData: nil)])
}

/*: Initializing
 ## Initialize the network weights
 
 The weights must be initialized to non-uniform values.  You can create your own initializer routine, or just let the network create random initial weights itself.
 Because the weights are initialized to random values, the author of this playground cant be sure what results you are getting!.  If something doesn't quite look like what is being talked about, force the playground to run again and see if you get results closer to what is usually expected
 */

//  Initialize the weights
network.initializeWeights(nil)


/*: Training
 ## Train the network
 
 We now train the network on the data.  This is generally done using a variation of stochastic gradient descent.  The data is processed in a series of packets called 'epochs'.  Each epoch uses a subset of the data.  We have a small data set, so we will train on all the data each epoch (epochSize = data.size).
 The number of epochs required varies.  Smaller networks require less training, while the requirement can go up exponentially with the network.  Of course, more training takes more time.
 The training rate parameter is the rate that changes are made to nodes' weights.  Increasing it can lower learning time, but also increases the odds of a weight 'jumping over' a particularly good solution because the learning rate made the change large.
 The weight decay parameter is a form of regularization.  Each training epoch the weights are multiplied by that amount.  This will decay unused weights, possibly improving generalization of the network.
 */
var numberOfEpochs = 500
if (exampleIndex >= 2) { numberOfEpochs = 50000 }
try network.classificationSGDBatchTrain(data, epochSize: data.size, epochCount : numberOfEpochs, trainingRate: 1.0, weightDecay: 1.0)


/*: Plotting
 ## Create a Data Plot
 
 AIToolbox includes a subclass of NSView that can be used to plot the data, regression functions, and classification areas.  We use one here to show the data
 
 There are different types of plot objects that can be added.  Data, classifier areas, axis labels, and legends will be used here.  Plot objects are drawn in the order added to the view.
 
 Start with creating the view
 */
let dataView = MLView(frame: CGRect(x: 0, y: 0, width: 480, height: 320))

//:  Create a classification data set plot item
let dataPlotItem = try MLViewClassificationDataSet(dataset: data)
//:  Set that item as the source of the initial scale of plot items.  Some plot items can be set to change the scale, but something needs to set the initial scale
dataView.setInitialScaleItem(dataPlotItem)
//:  Add a set of labels first (so they are underneath any other drawing).  Both X and Y axis labels
let axisLabels = MLViewAxisLabel(showX: true, showY: true)
dataView.addPlotItem(axisLabels)
//:  Add the classification region
let classifierArea = MLViewClassificationArea(classifier: network)
//:  You can adjust the size of each plot 'pixel' by setting the granularity.  Be warned, a 1 point area pixel can take some time to process all the data.  You can even go to a granularity of 0 to get a 0.5-point rectangle - individual pixels on a retina display - at a cost!
classifierArea.granularity = 4
dataView.addPlotItem(classifierArea)
//:  Add the data plot item now (so the data points are on top of the areas
dataView.addPlotItem(dataPlotItem)
//:  Create a legend
let legend = MLViewLegend(location: .upperRight, title: "Legend")
//:  Add the data point labels to the legend (they coincide with the area colors)
let dataLegends = MLLegendItem.createClassLegendArray("class ", classificationDataSet: dataPlotItem)
legend.addItems(dataLegends)
dataView.addPlotItem(legend)

//:  Finally, set the view to be drawn by the playground
PlaygroundPage.current.liveView = dataView

/*: Results
 ## Expected results
 
 Change the 'exampleIndex' value above to switch between the different experiments.
 
 ### Experiment 1
 The single-node network will likely have no problem finding a simple linear boundary separating the points.  The colored areas on the plot should include only dots of a similar color.
 
 ### Experiment 2
 The single-node network will not be able to find a line that can separate the now non-linearly-separable points.  A single node can only discriminate using a single hyperplane.
 
 ### Experiment 3
 The two-layer network can now create a two-dimensional structure to separate the points.  Based on initial conditions, this may be a really good decision boundary, or a poor one.  Re-run this one a couple of times to see what comes out.
 
 ### Experiment 4
 The points for class 1 are now in two groups, with class 0 points between.  The network creates a 'band' region joining the two groups.  However, this leaves a mis-classified point.
 
 ### Experiment 5
 The three-layer network, with enough nodes to create multiple non-linear areas and combine them, can solve the problem.  But it doesn't always.  Again, initial conditions play a big part of the result.  Sometimes it doesn't solve completely (stuck in a local minimum that doesn't separate the points), sometimes it creates multiple regions, and sometimes it creates one convoluted area that does contain all the class 1 points.
 */

