# Protocols
This section of the manual describes all the details of the Swift protocols used by the AIToolbox framework.


##MLDataSet
The MLDataSet protocol defines the required interface for the 'input' side of a data set.  The protocol contains the following required member variables:

var | Type | Access | Description
--- | ---- | ------ | -------
	dataType | DataSetType | get |  the type of data set.  This is a DataSetType enumeration that declares the type of output from this data set
	inputDimension | Int | get | the dimension of the input vector for the data set
	outputDimension | Int | get | the dimension of the output vector if the dataType is .regression or .realAndClass
	size | Int | get | the number of data points defined in the set
	optionalData | AnyObject? | get, set | an optional object that can be attached to the data set by an algorithm for additional data storage during computation
The MLDataSet protocol has only one defined function:
###getInput
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


##MLRegressionDataSet
The MLRegressionDataSet protocol is the required interface for a data set that will be used by regression algorithms.  It inherits the MLDataSet protocol, so all input requirements from that protocol are already defined.  The MLRegressionDataSet defines the following functions regarding the outputs of a data set:
	getOutput(_ index: Int) throws ->[Double] - this method gets the regression output vector of real values for the specified index.  It can throw a DataIndexError exception is thrown if the index is outside the valid range, or a DataTypeError exception if the data set is not set up for regression.
	setOutput(_ index: Int, newOutput : [Double]) throws - this method sets the regression output vector of real values for the specified index from the passed in vector.  It can throw a DataIndexError exception is thrown if the index is outside the valid range, or a DataTypeError exception if the data set is not set up for regression.

##MLClassificationDataSet
The MLClassificationDataSet protocol is the required interface for a data set that will be used by classification algorithms (unless it will also have an output vector of reals for other purposes).  It inherits the MLDataSet protocol, so all input requirements from that protocol are already defined.  The MLClassificationDataSet defines the following functions regarding the outputs of a data set:
	getClass(_ index: Int) throws ->Int - this method gets the classification output class identifier integer for the specified index.  It can throw a DataIndexError exception is thrown if the index is outside the valid range, or a DataTypeError exception if the data set is not set up for classification.
	setClass(_ index: Int, newClass : Int) throws - this method sets the classification output class identifier integer for the specified index from the passed in value.  It can throw a DataIndexError exception is thrown if the index is outside the valid range, or a DataTypeError exception if the data set is not set up for classification.
	
##MLCombinedDataSet
The MLCombinedDataSet protocol the required interface for a data set that will be used by classification algorithms that also have an output vector of reals for other purposes.  It inherits the MLRegressionDataSet  and MLClassificationDataSet  protocols, so all input and output requirements from those protocols are already defined.  No non-inherited functionality is defined by the protocol.


