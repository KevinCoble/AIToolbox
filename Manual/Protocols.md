# Protocols
This section of the manual describes all the details of the Swift protocols used by the AIToolbox framework.

##ConstraintProblemConstraint
The ConstraintProblemConstraint protocol defines the required interface for a class used as custom constraints in a constraint-propagation problem.  The protocol contains the following required member variable:

var | Type | Access | Description
--- | ---- | ------ | -------
	isSelfConstraint | Bool | get |  the flag indicating if the constraint is for the current node, or a connected node
The ConstraintProblemConstraint protocol has only one defined function:
####enforceConstraint
<table border="1">
<tr>
	<td>Template</td>
	<td>enforceConstraint(_ graphNodeList: [ConstraintProblemNode], forNodeIndex: Int) -> [EnforcedConstraint]</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method is called to enforce the constraint on the graph node given by the passed in index in the passed in graph node list.  The function should return a list of <strong>EnforcedConstraint</strong> structures, each giving a node that was changed and the domain index that was removed from that node</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>
		<table border="1">
		<tr>
		<th>name</th>
		<th>Type</th>
		<th>Description</th>
		</tr>
		<tr>
		<td>graphNodeList</td>
		<td>[ConstraintProblemNode]</td>
		<td>the array of graph nodes that make up the problem</td>
		</tr>
		<tr>
		<td>forNodeIndex</td>
		<td>Int</td>
		<td>index of of the graph node that the constraint is to be enforced on</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>[EnforcedConstraint] - the array of <strong>EnforcedConstraint</strong> structures that make up the graph change when the constraint is enforced, each giving a node that was changed and the domain index that was removed from that node</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

##MLDataSet
The MLDataSet protocol defines the required interface for the 'input' side of a data set.  The protocol contains the following required member variables:

var | Type | Access | Description
--- | ---- | ------ | -------
	dataType | DataSetType | get |  the type of data set.  This is a DataSetType enumeration that declares the type of output from this data set
	inputDimension | Int | get | the dimension of the input vector for the data set
	outputDimension | Int | get | the dimension of the output vector if the dataType is .regression or .realAndClass
	size | Int | get | the number of data points defined in the set
	optionalData | AnyObject? | get, set | an optional object that can be attached to the data set by an algorithm for additional data storage during computation
The MLDataSet protocol has only one required function:
####getInput
<table border="1">
<tr>
	<td>Template</td>
	<td>getInput(_ index: Int) throws ->[Double]</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method gets the input vector for the specified index</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>
		<table border="1">
		<tr>
		<th>name</th>
		<th>Type</th>
		<th>Description</th>
		</tr>
		<td>index</td>
		<td>Int</td>
		<td>index of input vector to get.  The index should be between zero and <strong>size</strong>-1</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>[Double] - The input vector for the specified index</td>
</tr>
<tr>
	<td>Throws</td>
	<td>
		<table border="1">
		<tr>
		<th>Error</th>
		<th>Description</th>
		</tr>
		<td>DataIndexError.negative</td>
		<td>thrown if the index is less than zero</td>
		</tr>
		</tr>
		<td>DataIndexError.indexAboveDataSetSize</td>
		<td>thrown if the index is outside the valid range</td>
		</tr>
		</table>
	</td>
</tr>
</table>

The MLDataSet protocol also has one function in an extension that is already filled out:
```swift
    public func getRandomIndexSet() -> [Int]
```
This function returns a random set if indices into the data set.  This array of random indices is useful for stochastic batch training routines.

##MLRegressionDataSet
The MLRegressionDataSet protocol is the required interface for a data set that will be used by regression algorithms.  It inherits the MLDataSet protocol, so all input requirements from that protocol are already defined.  The MLRegressionDataSet defines the following functions regarding the outputs of a data set:
####getOutput
<table border="1">
<tr>
	<td>Template</td>
	<td>getOutput(_ index: Int) throws ->[Double]</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method gets the output vector for the specified index</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>
		<table border="1">
		<tr>
		<th>name</th>
		<th>Type</th>
		<th>Description</th>
		</tr>
		<td>index</td>
		<td>Int</td>
		<td>index of output vector to get.  The index should be between zero and <strong>size</strong>-1</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>[Double] - The output vector for the specified index</td>
</tr>
<tr>
	<td>Throws</td>
	<td>
		<table border="1">
		<tr>
		<th>Error</th>
		<th>Description</th>
		</tr>
		<td>DataIndexError.negative</td>
		<td>thrown if the index is less than zero</td>
		</tr>
		</tr>
		<td>DataIndexError.indexAboveDataSetSize</td>
		<td>thrown if the index is outside the valid range</td>
		</tr>
		</tr>
		<td>DataTypeError.dataWrongForType</td>
		<td>thrown if the data set is not a regression data set</td>
		</tr>
		</table>
	</td>
</tr>
</table>

####setOutput
<table border="1">
<tr>
	<td>Template</td>
	<td>setOutput(_ index: Int, newOutput : [Double]) throws</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method sets the output vector for the specified index</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>
		<table border="1">
		<tr>
		<th>name</th>
		<th>Type</th>
		<th>Description</th>
		</tr>
		<td>index</td>
		<td>Int</td>
		<td>index of output vector to set.  The index should be between zero and <strong>size</strong>-1</td>
		</tr>
		</tr>
		<td>newOutput</td>
		<td>[Double]</td>
		<td>output vector to be set on the point indicated by the index parameter</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>None</td>
</tr>
<tr>
	<td>Throws</td>
	<td>
		<table border="1">
		<tr>
		<th>Error</th>
		<th>Description</th>
		</tr>
		<td>DataIndexError.negative</td>
		<td>thrown if the index is less than zero</td>
		</tr>
		</tr>
		<td>DataIndexError.indexAboveDataSetSize</td>
		<td>thrown if the index is outside the valid range</td>
		</tr>
		</tr>
		<td>DataTypeError.dataWrongForType</td>
		<td>thrown if the data set is not a regression data set</td>
		</tr>
		</table>
	</td>
</tr>
</table>

##MLClassificationDataSet
The MLClassificationDataSet protocol is the required interface for a data set that will be used by classification algorithms (unless it will also have an output vector of reals for other purposes).  It inherits the MLDataSet protocol, so all input requirements from that protocol are already defined.  The MLClassificationDataSet defines the following functions regarding the outputs of a data set:
####getClass
<table border="1">
<tr>
	<td>Template</td>
	<td>getClass(_ index: Int) throws ->Int</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method gets the output class label for the specified index</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>
		<table border="1">
		<tr>
		<th>name</th>
		<th>Type</th>
		<th>Description</th>
		</tr>
		<td>index</td>
		<td>Int</td>
		<td>index of output class label to get.  The index should be between zero and <strong>size</strong>-1</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>Int - The class label for the specified index</td>
</tr>
<tr>
	<td>Throws</td>
	<td>
		<table border="1">
		<tr>
		<th>Error</th>
		<th>Description</th>
		</tr>
		<td>DataIndexError.negative</td>
		<td>thrown if the index is less than zero</td>
		</tr>
		</tr>
		<td>DataIndexError.indexAboveDataSetSize</td>
		<td>thrown if the index is outside the valid range</td>
		</tr>
		</tr>
		<td>DataTypeError.dataWrongForType</td>
		<td>thrown if the data set is not a classification data set</td>
		</tr>
		</table>
	</td>
</tr>
</table>

####setClass
<table border="1">
<tr>
	<td>Template</td>
	<td>setClass(_ index: Int, newClass : Int) throws</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method sets the output vector for the specified index</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>
		<table border="1">
		<tr>
		<th>name</th>
		<th>Type</th>
		<th>Description</th>
		</tr>
		<td>index</td>
		<td>Int</td>
		<td>index of the class label to set.  The index should be between zero and <strong>size</strong>-1</td>
		</tr>
		</tr>
		<td>newClass</td>
		<td>Int</td>
		<td>class label to be set on the point indicated by the index parameter</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>None</td>
</tr>
<tr>
	<td>Throws</td>
	<td>
		<table border="1">
		<tr>
		<th>Error</th>
		<th>Description</th>
		</tr>
		<td>DataIndexError.negative</td>
		<td>thrown if the index is less than zero</td>
		</tr>
		</tr>
		<td>DataIndexError.indexAboveDataSetSize</td>
		<td>thrown if the index is outside the valid range</td>
		</tr>
		</tr>
		<td>DataTypeError.dataWrongForType</td>
		<td>thrown if the data set is not a classification data set</td>
		</tr>
		</table>
	</td>
</tr>
</table>
	
##MLCombinedDataSet
The MLCombinedDataSet protocol is the required interface for a data set that will be used by classification algorithms that also have an output vector of reals for other purposes.  It inherits the MLRegressionDataSet  and MLClassificationDataSet  protocols, so all input and output requirements from those protocols are already defined.  No non-inherited functionality is defined by the protocol.

##Regressor
The Regressor protocol is an interface used by the majority of regression models in AIToolbox.  Using the protocol allows all the models to behave the same, allowing them to use **Validation** methods for parameter tuning and model selection without custom code.  The protocol has the following required functions:

####getInputDimension
<table border="1">
<tr>
	<td>Template</td>
	<td>getInputDimension() -> Int</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method gets the required dimension of the input vectors used by the model</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>
		None
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>Int - The dimension of the input vector used by the model</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

####getOutputDimension
<table border="1">
<tr>
	<td>Template</td>
	<td>getOutputDimension() -> Int</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method gets the required dimension of the output vectors used by the model</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>
		None
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>Int - The dimension of the output vector used by the model</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

####getParameterDimension
<table border="1">
<tr>
	<td>Template</td>
	<td>getParameterDimension() -> Int</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method gets the number of parameters used by the model</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>
		None
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>Int - The number of parameters used by the model</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

####setParameters
<table border="1">
<tr>
	<td>Template</td>
	<td>setParameters(_ parameters: [Double]) throws</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method will set the parameters of the model from the passed in array of values</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>
		<table border="1">
		<tr>
		<th>name</th>
		<th>Type</th>
		<th>Description</th>
		</tr>
		<td>parameters</td>
		<td>[Double]</td>
		<td>The values to set the parameters to.  Should be sized based on the <strong>getParameterDimension</strong> result</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>None</td>
</tr>
<tr>
	<td>Throws</td>
	<td>
		<table border="1">
		<tr>
		<th>Error</th>
		<th>Description</th>
		</tr>
		<td>MachineLearningError.notEnoughData</td>
		<td>thrown if the size of the passed in array is smaller than that required.  Use the <strong>getParameterDimension</strong> method to get the required size</td>
		</tr>
		</table>
	</td>
</tr>
</table>

####setCustomInitializer
<table border="1">
<tr>
	<td>Template</td>
	<td>setCustomInitializer(_ function: ((_ trainData: MLDataSet)->[Double])!)</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method will set a function that will be called on parameter initialization, with the initial training data set as the parameter.  If nil [the default state], the parameters will be set to random values.  If this function is not nil, the results of this function is used to initialize the parameters</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>
		<table border="1">
		<tr>
		<th>name</th>
		<th>Type</th>
		<th>Description</th>
		</tr>
		<td>function</td>
		<td>((_ trainData: MLDataSet)->[Double])!</td>
		<td>The function that will get the initial training data set passed to it and should return the values to be used to initialize the parameters</strong> result</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>None</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

####getParameters
<table border="1">
<tr>
	<td>Template</td>
	<td>getParameters() throws -> [Double]</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method will get the parameters of the model.  The model may need to be trained first</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>
		None
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>[Double] - the parameters of the model</td>
</tr>
<tr>
	<td>Throws</td>
	<td>
		<table border="1">
		<tr>
		<th>Error</th>
		<th>Description</th>
		</tr>
		<td>MachineLearningError.notTrained</td>
		<td>thrown if the model has not been trained, and is needed by the method before parameters are created</td>
		</tr>
		</table>
	</td>
</tr>
</table>

####trainRegressor
<table border="1">
<tr>
	<td>Template</td>
	<td>trainRegressor(_ trainData: MLRegressionDataSet) throws</td>
</tr>
<tr>
	<td>Description</td>
	<td>This trains the regression model on the data set passed in.  The model parameters will be initialized by the method before training begins.</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>
		<table border="1">
		<tr>
		<th>name</th>
		<th>Type</th>
		<th>Description</th>
		</tr>
		<td>trainData</td>
		<td>MLRegressionDataSet</td>
		<td>The data set to train on.  The data set dimensions (input and output) must match that of the model</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>None</td>
</tr>
<tr>
	<td>Throws</td>
	<td>
		<table border="1">
		<tr>
		<th>Error</th>
		<th>Description</th>
		</tr>
		<tr>
		<td>MachineLearningError.dataNotRegression</td>
		<td>thrown if the data set used for training is not a regression set</td>
		</tr>
		<tr>
		<td>MachineLearningError.dataWrongDimension</td>
		<td>thrown if the data set used for training does not match the model</td>
		</tr>
		<tr>
		<td>MachineLearningError.notEnoughData</td>
		<td>thrown if the data set used for training does not have enough data to train the model (usually the number of parameters in the model exceeds the data set size)</td>
		</tr>
		</table>
		Additional exceptions may be thrown by individual model classes
	</td>
</tr>
</table>

####continueTrainingRegressor
<table border="1">
<tr>
	<td>Template</td>
	<td>continueTrainingRegressor(_ trainData: MLRegressionDataSet) throws</td>
</tr>
<tr>
	<td>Description</td>
	<td>This trains the regression model on the data set passed in.  The model parameters are not initialized first, so training continues with the current parameter set.  Not all regression models support this.</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>
		<table border="1">
		<tr>
		<th>name</th>
		<th>Type</th>
		<th>Description</th>
		</tr>
		<td>trainData</td>
		<td>MLRegressionDataSet</td>
		<td>The data set to train on.  The data set dimensions (input and output) must match that of the model</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>None</td>
</tr>
<tr>
	<td>Throws</td>
	<td>
		<table border="1">
		<tr>
		<th>Error</th>
		<th>Description</th>
		</tr>
		<tr>
		<td>MachineLearningError.dataNotRegression</td>
		<td>thrown if the data set used for training is not a regression set</td>
		</tr>
		<tr>
		<td>MachineLearningError.dataWrongDimension</td>
		<td>thrown if the data set used for training does not match the model</td>
		</tr>
		<tr>
		<td>MachineLearningError.continuationNotSupported</td>
		<td>thrown if the regression model does not support training continuation</td>
		</tr>
		</table>
		Additional exceptions may be thrown by individual model classes
	</td>
</tr>
</table>

####predictOne
<table border="1">
<tr>
	<td>Template</td>
	<td>predictOne(_ inputs: [Double]) throws ->[Double]</td>
</tr>
<tr>
	<td>Description</td>
	<td>This gets the regression results for a single input vector.  The input vector size must match the model input size.</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>
		<table border="1">
		<tr>
		<th>name</th>
		<th>Type</th>
		<th>Description</th>
		</tr>
		<td>inputs</td>
		<td>[Double]</td>
		<td>The input vector to get the regression values for</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>[Double] - the resulting regression values</td>
</tr>
<tr>
	<td>Throws</td>
	<td>
		<table border="1">
		<tr>
		<th>Error</th>
		<th>Description</th>
		</tr>
		<tr>
		<td>DataTypeError.wrongDimensionOnInput</td>
		<td>thrown if the input vector dimension does not match the model</td>
		</tr>
		<tr>
		<td>MachineLearningError.notTrained</td>
		<td>thrown if the model has not yet been trained</td>
		</tr>
		</table>
		Additional exceptions may be thrown by individual model classes
	</td>
</tr>
</table>

####predict
<table border="1">
<tr>
	<td>Template</td>
	<td>predict(_ testData: MLRegressionDataSet) throws</td>
</tr>
<tr>
	<td>Description</td>
	<td>This gets the regression results for a all points in a data set.  The data set type and dimensions must match the model requirements.</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>
		<table border="1">
		<tr>
		<th>name</th>
		<th>Type</th>
		<th>Description</th>
		</tr>
		<td>testData</td>
		<td>MLRegressionDataSet</td>
		<td>The data set to get results for.  The data set dimensions (input and output) must match that of the model</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>(the data set class passed in is modified to have the output results)</td>
</tr>
<tr>
	<td>Throws</td>
	<td>
		<table border="1">
		<tr>
		<th>Error</th>
		<th>Description</th>
		</tr>
		<tr>
		<td>MachineLearningError.dataNotRegression</td>
		<td>thrown if the data set used for training is not a regression set</td>
		</tr>
		<tr>
		<td>MachineLearningError.dataWrongDimension</td>
		<td>thrown if the data set used for training does not match the model</td>
		</tr>
		<tr>
		<td>MachineLearningError.notTrained</td>
		<td>thrown if the model has not yet been trained</td>
		</tr>
		</table>
		Additional exceptions may be thrown by individual model classes
	</td>
</tr>
</table>

    


