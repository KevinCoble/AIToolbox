/*: Introduction
 # Support Vector Machines
 
 This playground shows how the basics of Support Vector Machines, and shows how to use the AIToolbox framework to generate training data, train an svm model, and plot the results
 
 We start with the required import statements.  If this was your own program, rather than a playground, you would also add an 'import AIToolbox' line.
 */
import UIKit
import PlaygroundSupport

/*: Data Set
 ## Create a Classification DataSet
 
 We need a data set to train our Support Vector Machine on
 
 DataSets in AIToolbox come in two types, Regression and Classification.  Regression is where you get a value (or array of values if the output dimension is >1) for your inputs.  Classification is when you get an integer 'class label' based on the inputs.  Regression is used to do things like 'predict time till failure', while classification is used for tasks like 'is that tumor malignant or benign'
 
 This is a Support Vector Machine, so our data set will be the 'Classification' type, with an input input vector sized for two variables (the X and Y axis on our plot later).
 */
let data = DataSet(dataType: .classification, inputDimension: 2, outputDimension: 1)
//:  Add a few data points to the data set.  Note the 'output' category is a single integer.  This is the 'class label' for the point.  All points with the same label are assumed to be in the same class.
try data.addDataPoint(input: [0.2, 0.9], dataClass:0)
try data.addDataPoint(input: [0.8, 0.3], dataClass:0)
try data.addDataPoint(input: [0.5, 0.6], dataClass:0)
try data.addDataPoint(input: [0.2, 0.7], dataClass:1)
try data.addDataPoint(input: [0.2, 0.3], dataClass:1)
try data.addDataPoint(input: [0.4, 0.5], dataClass:1)
try data.addDataPoint(input: [0.5, 0.4], dataClass:2)
try data.addDataPoint(input: [0.3, 0.2], dataClass:2)
try data.addDataPoint(input: [0.7, 0.2], dataClass:2)

/*: Kernel
 ## Define the Kernel function to use
 
 A Support Vector Machine uses a 'kernel' function to replace the dot-product of two input vectors.  Doing this allows the input space to easily be converted to a higher-dimension space, before calculating linearly seperable boundaries.  If just the dot-product is used, the points must be seperable by a hyperplane.  Using a kernel, parabaloids, and radial functions, etc. can be used instead.  It is even possible to use a Taylor-series expansion term, for an infinite dimensional input space!
 
 There are four different types of kernels available in AIToolbox, linear (dot-product), polynomial, radial basis functions, and sigmoids.  Some of the kernel types take additional parameters, like the degree of a polynomial kernel
 
 Below are two examples of kernel functions.  The first is for a radial basis function, the second for a 3rd degree polynomial
 */
let kernelSettings = KernelParameters(type: .radialBasisFunction, degree: 0, gamma: 0.5, coef0: 0.0)
//let kernelSettings = KernelParameters(type: .Polynomial, degree: 3, gamma: 0.5, coef0: 0.0)

/*: SVM
 ## Create a Support Vector Machine
 
 Once you have a kernel, use it to create a Support Vector Machine model
 
 There are several different types of solutions for an SVM, classification, regression, and multiple solutions types for each.  We will use standard classification.
 */

let svm = SVMModel(problemType: .c_SVM_Classification, kernelSettings: kernelSettings)
/*: SVM
 ### Adjusting the Cost
 
 Not all data can be seperated by the kernel functions.  If you leave strict classification, the solution will fail in these cases.  To alleviate this problem, you can assign a cost to mis-classified data (it is possible to adjust the cost for each different class even).  This cost value is the penalty for mis-classification.  If the number is low, errors are tolerated more.  If high, more effort is taken to find a seperation
 
 Below are two possible cost values.  See what both do to the green dots.
 */
svm.Cost = 30.0     //  Set the cost of a misclasification
//svm.Cost = 1.0     //  Set the cost of a misclasification

/*: Training
 ## 'Train' the model
 
 Now that we have everything set up, have the SVM Model 'train' - or find the best solution it can.
 */
svm.train(data)
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
let classifierArea = MLViewClassificationArea(classifier: svm)
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

