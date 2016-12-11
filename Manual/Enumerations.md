# Enumerations
This section of the manual describes all the details of the Swift enumerations used by the AIToolbox framework.

##Convolution2DMatrix
The type of convolution done by a Convolution2D operator
case |  Description
--- | ---- 
    verticalEdge3 | A standard Sobel filter for vertical edges in a 3x3 matrix
    horizontalEdge3 | A standard Sobel filter for horizontal edges in a 3x3 matrix
    custom3 | A custom 3x3 matrix
    learnable3 | A custom 3x3 matrix that can be learned from data

##DataSetType
Defines the type of the output of a data set

case |  Description
--- | ---- 
		regression | for regression data sets (array of real numbers output)
		classification | for classification data sets (single integer output)
		realAndClass | for classification data sets that require real number outputs (like probabilities of each class)
		
##DeepNetworkOperatorType
The type of operator in a deep-network channel operation list

case |  Description
--- | ---- 
    convolution2DOperation | Convolve a 2D matrix with a convolution matrix
    poolingOperation | Pool a matrix (any dimension) into a smaller one
    feedForwardNetOperation | Feed the incoming data into a neural network layer

##MLPlotSymbolShape
The shape of the symbol to be used to identify a data point on an MLView

case |  Description
--- | ---- 
    circle | circle
    rectangle | rectangle
    plus | plus sign
    minus | minus sign

##MLViewAxisSource
The source of values for an MLView axis

case |  Description
--- | ---- 
    dataInput | the input vector value at the specified index is used
    dataOutput | the output vector value at the specified index is used
    classLabel | the class label is used.  The index parameter is ignored

##MLViewLegendLocation
The location of a legend object in an MLView

case |  Description
--- | ---- 
    upperLeft | the upper-left corner of the view
    upperRight | the upper-right corner of the view
    lowerLeft | the lower-left corner of the view
    lowerRight | the lower-right corner of the view
    custom | a custom location specified by the caller

##NeuronLayerType
Defines the type of the layer in a neural network.  Networks containing recurrent style layers should be trained with sequences, rather than individual data.

case |  Description
--- | ---- 
    simpleFeedForwardWithNodes | a feed-forward layer that has each node as an individual class.  Easier to follow internally, but slower
    simpleFeedForward | a feed-forward layer that has all the weights in one matrix.  Faster than individual nodes
    simpleRecurrentWithNodes | a recurrent layer (previous outputs are also used as additional inputs) that has each node as an individual class.  Easier to follow internally, but slower
    simpleRecurrent | a recurrent layer (previous outputs are also used as additional inputs) that has has all the weights in two matrices.  Faster than individual nodes
    lstm | a long-short-term-memory layer (recurrent with flow gates)

##NeuralActivationFunction
Defines the function used to 'squash' the results of the dot-product of the inputs with the node's weights.

case |  Description
--- | ---- 
none | no change is made to the result of initial dot product result
hyperbolicTangent | the tanh function, which limits the result to the range [-1, 1] is used
sigmoid | the logistic function [1/(1+exp(-x)] is used, which limits the range to [0, 1]
sigmoidWithCrossEntropy | the sigmoid function is used, but cross-entropy loss function is preferred (this is not yet implemented fully, and may be removed or changed later)
rectifiedLinear | values below zero are clipped to zero, else no change to the value
softSign | x / (1.0 + abs(x))
softMax | Only valid on output (last) layer.  The function  used is just exp(x), but each value is normalized by the sum of all the outputs of the nodes in the layer.  This gives the output vector the look of a probability distribution (all values in the range [0,1], with the total equal to 1)

##NeuralWeightUpdateMethod
The method to use when updating the weights from (possibly accumulated) gradients with respect to the error term.

case |  Description
--- | ---- 
    normal | normal update (W -= learning_rate * gradient).  No parameter required.
    rmsProp	| root-mean-square propogation (rmsprop_cache = decay_rate * rmsprop_cache + (1 - decay_rate) * gradient², W -= learning_rate * gradient / (sqrt(rmsprop_cache) + 1e-5) ).  Requires decay rate as parameter.

##NonLinearRegressionConvergenceType
The type of criteria used to determine if a non-linear regression model has converged

case |  Description
--- | ---- 
    smallGradient | the gradient of the model becomes small
    smallAverageLoss	| the average loss of the training data becomes small
    smallParameterChange | the changes to the parameters becomes small.  This is the only option for non-linear regression models using parameterDelta for the solution type

##NonLinearRegressionType
The solution method for a non-linear regression model

case |  Description
--- | ---- 
    parameterDelta | small changes are made to the parameters of the model, and kept if the total error decreases
    sgd | stochastic gradient descent
    gaussNewton | a Gauss-Newton method

##PoolingType
The type of operation used in a deep-network pooling operator

case |  Description
--- | ---- 
    average | average of the matrix volume being pooled
    minimum | minimum of the matrix volume being pooled
    maximum | maximum of the matrix volume being pooled

##StandardConstraintType
The type of constraint in a standard constraint propagation problem, when not using custom constraint functions.

case |  Description
--- | ---- 
    cantBeSameValueInOtherNode | All nodes that connect to the current node in the graph cannot have the same value as the current node
    mustBeSameValueInOtherNode | All nodes that connect to the current node in the graph must have the same value as the current node
    cantBeValue | All nodes that connect to the current node in the graph cannot have the value specified in the constraing
    mustBeGreaterThanOtherNode | All nodes that connect to the current node in the graph must have a larger value than the current node
    mustBeLessThanOtherNode | All nodes that connect to the current node in the graph must have a lesser value than the current node
    
##SubtermFunction
 The type of function in a linear regression model subterm

case |  Description
--- | ---- 
    none | no function is added to the subterm
    naturalExponent | the subterm is placed inside a natural exponent function
    sine | the subterm is placed inside a sine function
    cosine | the subterm is placed inside a cosine function
    log | the subterm is placed inside a natural logarithm function
    power | the subterm is the power value of the term raised to input

##SVMKernelType
The type of an SVM kernel

case |  Description
--- | ---- 
    linear | a linear kernel
    polynomial | a polynomial kernel of the specified polygon order
    radialBasisFunction | radial basis functions with the given gamma
    sigmoid | sigmoid function kernel
    precomputed | (not implemented at this time)
    
##SVMType
The type of SVM model.  These came from the LIBSVM port

case |  Description
--- | ---- 
    c_SVM_Classification | standard classification
    ν_SVM_Classification | ν solution method classification
    oneClassSVM | single class classification
    ϵSVMRegression | standard regression
    νSVMRegression | ν solution method regression

# Thrown Error Enumerations
##DataIndexError
Inherits from **Error**.  Defines exceptions thrown for errors in data indexing values

case |  Description
--- | ---- 
		negative | the index for the data point was negative
		indexAboveDimension | the index for a sub-element was above the dimension bounds for the array
		indexAboveDataSetSize | the index was greater than or equal to the number of points in the data set
		
##DataTypeError
Inherits from **Error**.  Defines exceptions thrown for errors with the type of data being added to a data set

case |  Description
--- | ---- 
		invalidDatatype | the data type of the data set is not correct for the operation requested (i.e. grouping classes on a regression data set)
		dataWrongForType | the output data being added to a data set is not correct for the type the data set was created for
		wrongDimensionOnInput | the vector passed as an input vector to a data set does not match the input dimension the data set was created for
		wrongDimensionOnOutput | the vector passed as an output vector to a regression data set does not match the output dimension the data set was created for

##GaussianError
Inherits from **Error**.  Defines exceptions thrown for errors using gaussian (mostly muti-variate gaussians) objects

case |  Description
--- | ---- 
    dimensionError | the input vector did not match the dimension of the gaussian
    zeroInVariance | the variance matrix had a zero determinant, causing a mathematical error
    inverseError | the variance matrix could not be inverted
    badVarianceValue | attempt to set a bad value (negative) for a variance, or set a variance outside the range of a multi-variate gaussian variance matrix dimensions
    diagonalCovarianceOnly | attempt to set a variance value off the diagonal of a diagonal-only multi-variate gaussian
    errorInSVDParameters | error in parameters passed to the eigen-value calculation routines in internal math
    svdDidNotConverge | the eigen-value calculation routines in internal math did not converge
    
##LinearRegressionError
Inherits from **Error**.  Defines exceptions thrown for errors using linear regression models

case |  Description
--- | ---- 
    modelExceedsInputDimension | the linear regression model has more parameters than the input vector dimension
    matrixSolutionError | the matrix in the regression calculation could not be inverted
    negativeInLogOrPower | negative values in the input conflicted with logarithm or power functions in the model
    divideByZero | the regression model attempted to divide by zero
    
##MachineLearningError
Inherits from **Error**.  Defines exceptions thrown for errors using many machine learning routines

case |  Description
--- | ---- 
    dataNotRegression | regression data was needed, but classification data was provided
    dataNotClassification | classification data was needed, but regression data was provided
    dataWrongDimension | the data passed in was the wrong dimension for the model
    notEnoughData | not enough data was provided for the operation (for example, less data points than parameters in a regression learning)
    modelNotRegression | the function is for use in regression models only 
    modelNotClassification | the function is for use in classification models only 
    notTrained | attempt to use a model that has not yet been trained
    initializationError | there was an error initializeing the model
    didNotConverge | the model did not converge on a solution
    continuationNotSupported | the model only supports one-time training
    operationTimeout | the operation took longer than expected and was terminated early
    continueTrainingClassesNotSame | the classification data passed into a training continuation function contained classes that were not seen with the initial training

##MDPErrors
Inherits from **Error**.  Defines exceptions thrown for errors using many Markov-Decision-Process routines

case |  Description
--- | ---- 
    mdpNotSolved | attempt to get an action for an MDP that has not yet been solved
    failedSolving | the MDP failed in one-step solving
    errorCreatingSampleSet | there was an internal error creating sample episodes
    errorCreatingSampleTargetValues | there was an internal error creating sample episodes
    modelInputDimensionError | the learning model specified uses an input dimension different than the data
    modelOutputDimensionError | the learning model specified has an output dimension different than the data
    invalidState | the state specified is outside the MDP state set
    noDataForState | no data has been supplied for the state, so Q or V values can not yet be determined

##MixtureOfGaussianError
Inherits from **Error**.  Defines exceptions thrown for errors using a MixtureOfGaussian model

case |  Description
--- | ---- 
    kMeansFailed | the K-Means grouping of the data failed
    
##MLViewError
Inherits from **Error**.  Defines exceptions thrown for errors using a MLView view

case |  Description
--- | ---- 
    dataSetNotRegression | attempted to add a regression data set object, but the data set was not of the regression type
    dataSetNotClassification | attempted to add a classification data set object, but the data set was not of the classification type
    inputVectorNotOfCorrectSize | the input vector passed did not match the size of data being displayed
    inputIndexOutsideOfRange | the index to be displayed as an axis is outside of the input vector dimension
    outputIndexOutsideOfRange | the index to be displayed as an axis is outside of the output vector dimension
    allClassificationLabelsNotCovered | the classification label object did not cover all known class labels
    
##NeuralNetworkError
Inherits from **Error**.  Defines exceptions thrown for errors using a NeuralNetwork that are not covered by general machine learning errors

case |  Description
--- | ---- 
    expectedOutputNotSet | the expected output was not set before trying to gradient check a network

##NonLinearRegressionError
Inherits from **Error**.  Defines exceptions thrown for errors using a NonLinearRegression that are not covered by general machine learning errors

case |  Description
--- | ---- 
    convergenceTypeNotAllowed | the selected convergence type does not match the selected solution type
    matrixSolutionError | an error occurred solving the matrix equations for the solution method (Gauss-Newton)
    
##PCAWriteErrors
Inherits from **Error**.  Defines exceptions thrown for errors writing Primary Component Analysis parameters to a file

case |  Description
--- | ---- 
failedWriting | error writing the parameters
    
##SVMWriteErrors
Inherits from **Error**.  Defines exceptions thrown for errors writing Support Vector Machine parameters to a file

case |  Description
--- | ---- 
failedWriting | error writing the parameters

##ValidationError
Inherits from **Error**.  Defines exceptions thrown for errors when using a validation set object

case |  Description
--- | ---- 
    computationError | overflow on the math determining relative loss on each model being validated



