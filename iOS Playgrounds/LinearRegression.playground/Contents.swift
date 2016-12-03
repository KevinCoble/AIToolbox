/*: Introduction
 # Linear Regression
 
 This playground shows how the basics of Linear Regression, and shows how to use the AIToolbox framework to generate test data, train a linear regression model, and plot the results
 
 We start with the required import statements.  If this was your own program, rather than a playground, you would also add an 'import AIToolbox' line.
 */
import UIKit
import PlaygroundSupport

/*: Target Function
 ## Create a Target function
 
 First we need some data.  You can hard-code data points, or you can generate them from a function.  This playground will generate them from a function and add noise to the data
 
 Linear regression input data can be of any dimension.  Since we want to plot the data, we will use a 1 dimensional input (and output) for now
 
 We will use a linear regression model to generate the data.  Set the 'targetOrder' to the maximum power of the polynomial.
 */
let targetOrder = 2     //  Between 1 and 5
/*:
 The target function is created with the input and output size.  We will use a convenience initializer that takes the polynomial order and creates a linear function with no cross terms (y = A, y = A + Bx, y = A + Bx + Cx², etc.
 */
let target = LinearRegressionModel(inputSize: 1, outputSize: 1, polygonOrder: targetOrder)
/*:
 Next we create some parameters (this is the target function.  For the hypothesis function, we 'learn' these parameters)
 Start with an array of reasonable values, and add some gaussian noise to make different curves for each run.  Set the values as the parameters of the target function
 */
var targetParameters : [Double] = [10.0, 0.5, 0.25, -0.125, 0.0675, -0.038]
for index in 0...targetOrder {
    targetParameters[index] += Gaussian.gaussianRandom(0.0, standardDeviation: targetParameters[index])
}
try target.setParameters(targetParameters)
targetParameters        //  Show target parameters to compare with hypothesis parameters later

/*: Data Set
 ## Create a Regression DataSet
 
 Now that we have a target function, we need to get some data from it
 
 DataSets in AIToolbox come in two types, Regression and Classification.  Regression is where you get a value (or array of values if the output dimension is >1) for your inputs.  Classification is when you get an integer 'class label' based on the inputs.  Regression is used to do things like 'predict time till failure', while classification is used for tasks like 'is that tumor malignant or benign'
 
 This is Linear Regression, so our data set will be the 'Regression type, with the same input and output dimensions as our target function.
 */
let data = DataSet(dataType: .regression, inputDimension: 1, outputDimension: 1)
/*:
 We are going to use the target function to create points, and add noise
 set numPoints to the number of points to create, and set the standard deviation of the gaussian noise
 */

let numPoints = 100      //  Number of points to fit
let standardDeviation = 1.0     //  Amount of noise in target line

/*:
 The following code creates the points and adds them to the data set.
 We are keeping the input data in the range of 0-10, but that could be changed if you wanted.
 
 Note that the 'addDataPoint' function is preceded by a 'try' operator.  Many AIToolbox routines can throw errors if dimensions don't match
 */
for index in 0..<numPoints {
    let x = Double(index) * 10.0 / Double(numPoints)
    let targetValue = try target.predictOne([x])
    let y = targetValue[0] + Gaussian.gaussianRandom(0.0, standardDeviation: standardDeviation)
    try data.addDataPoint(input: [x], output:[y])
}


/*: Hypothesis
 ## Create a Regression Hypothesis Model
 
 The Hypothesis Model is the form of the function to 'learn' the parameters for.  Linear regression cannot create a function from the data, you specify the function and linear regression finds the best parameters to match the data to that function
 
 We will again use the convenience initializer to create a polygon model of a specific order.  The polygon order does NOT have to match that of the target function/data.  In fact, having a complicated hypothesis function with a small amount of data leads to 'overfitting'.  Note that the input and output dimensions have to match our data.
 */
let hypothesisOrder = 2     //  Polygon order for the model
let lr = LinearRegressionModel(inputSize: 1, outputSize: 1, polygonOrder: hypothesisOrder)


/*: Training
 ## 'Train' the model
 
 All regressors in AIToolbox, linear or not, have a 'trainRegressor' function that takes a Regression data set.  This function asks the hypothesis to find the best parameters for the data set passed in.
 
 Note the 'try' again.  Don't attempt to train a hypothesis with data of a different dimension.
 */
try lr.trainRegressor(data)
let parameters = try lr.getParameters()

/*: Plotting
 ## Create a Data Plot
 
 AIToolbox includes a subclass of NSView that can be used to plot the data and the functions.  We use one here to show the data
 
 There are different types of plot objects that can be added.  Data, functions, axis labels, and legends will be used here.  Plot objects are drawn in the order added to the view.
 
 Start with creating the view
 */
let dataView = MLView(frame: CGRect(x: 0, y: 0, width: 480, height: 320))
//:  Create a symbol object for plotting the data set - a green circle
let dataPlotSymbol = MLPlotSymbol(color: UIColor.green, symbolShape: .circle, symbolSize: 7.0)
//:  Create a regression data set plot item
let dataPlotItem = try MLViewRegressionDataSet(dataset: data, symbol: dataPlotSymbol)
//:  Set that item as the source of the initial scale of plot items.  Some plot items can be set to change the scale, but something needs to set the initial scale
dataView.setInitialScaleItem(dataPlotItem)
//:  Add a set of labels first (so they are underneath any other drawing).  Both X and Y axis labels
let axisLabels = MLViewAxisLabel(showX: true, showY: true)
dataView.addPlotItem(axisLabels)
//:  Create a regression data line item for the target function, blue
let targetLine = MLViewRegressionLine(regressor: target, color: UIColor.blue)
dataView.addPlotItem(targetLine)
//:  Create a regression data line item for the hypothes function, red
let hypothesisLine = MLViewRegressionLine(regressor: lr, color: UIColor.red)
dataView.addPlotItem(hypothesisLine)
//:  Add the data plot item now (so the data points are on top of the lines - reorder if you want!)
dataView.addPlotItem(dataPlotItem)
//:  Create a legend
let legend = MLViewLegend(location: .lowerRight, title: "Legend")
//:  Add the target line to the legend
let targetLegend = MLLegendItem(label: "target", regressionLinePlotItem: targetLine)
legend.addItem(targetLegend)
//:  Add the data set to the legend
let dataLegend = MLLegendItem(label: "data", dataSetPlotItem: dataPlotItem)
legend.addItem(dataLegend)
//:  Add the hypothesis line to the legend
let lineLegend = MLLegendItem(label: "hypothesis", regressionLinePlotItem: hypothesisLine)
legend.addItem(lineLegend)
dataView.addPlotItem(legend)

//:  Finally, set the view to be drawn by the playground
PlaygroundPage.current.liveView = dataView


