# Structures

##DeepChannelSize
The DeepChannelSize structure is used to define the size of data being passed between channels within the layers of a deep network.  The structure contains three variables:

var | Type | Access | Description
--- | ---- | ------ | -------
	numDimensions | Int | get |  the number of dimensions in the data.  Usually between 1 and 4, inclusive
	dimensions | [Int] | get, set | an array containing the size of each dimension.  The array should have numDimensions elements, with each element value 1 or greater

The DeepChannelSize structure has one function used to get a string representation:
####asString
<table border="1">
<tr>
	<td>Template</td>
	<td>asString() ->String</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method returns a string representation of the size array</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>None</td>
</tr>
<tr>
	<td>Output</td>
	<td>[String] - the string representation, of the form '[x, y, z]' (example with 3 dimensions)</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

##DeepNetworkInput
The DeepNetworkInput structure defines an input source for a deep-network.  Input sources (and channels) in a deep-network are labeled with strings, so that items further on in the network can reference them.  The DeepNetworkInput holds the floating input values, and a size structure that indicates how the data should be used.  The structure conforms to the **MLPersistence** protocol.  The structure has the following variables:

var | Type | Access | Description
--- | ---- | ------ | -------
	inputID | String | get | the string identifier for the input source
	size | DeepChannelSize | get | the size and dimensionality of the input data
	values | [Float] | get, set | the input values.  The number of elements in this array must batch the size.totalSize value

The DeepNetworkInput structure has only one other function, an initializer:

####init
<table border="1">
<tr>
	<td>Template</td>
	<td>init(inputID: String, size: DeepChannelSize, values: [Float])</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method creates a DeepNetworkInput with all fields initialized</td>
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
		<td>inputID</td>
		<td>String</td>
		<td>the identifier for the input data</td>
		</tr>
		<tr>
		<td>size</td>
		<td>DeepChannelSize</td>
		<td>the dimension of the data</td>
		</tr>
		<tr>
		<td>values</td>
		<td>[Float]</td>
		<td>the input values</td>
		</tr>
		</table>
</td>
</tr>
<tr>
	<td>Output</td>
	<td>N/A (constructor)</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

##EnforcedConstraint
The EnforcedConstraint structure is used by classes conforming to the **ConstraintProblemConstraint** protocol to return the results of enforcing a constraint on a graph node.  An array of these structures should be returned by the **enforceConstraint** method of the protocol.  The structure just has two variables and no methods:

var | Type | Access | Description
--- | ---- | ------ | -------
	nodeAffected | ConstraintProblemNode | get, set |  the node that was affected by enforcing the constraint
	domainIndexRemoved | Int | get, set | the constraint domain index that was removed from the affected node.
	
##KernelParameters
The KernelParameters structure is used to define the kernel used in an SVM solution.  The structure has the following four elements:

var | Type | Access | Description
--- | ---- | ------ | -------
	type | SVMKernelType | get |  the type of kernel equation used
	degree | Int | get |  the degree of polynomial if the 'type' member is .polynomial.  All other types ignore this value
	gamma | Double | get |  the coefficient for the dot-product of the input vectors for a polynomial, the negative power for the radial basis function, or the coefficient for the dot-product of the input vectors used inside the hyperbolic tangent function for sigmoid kernels.  All other types ignore this value
	coef0 | Double | get |  the bias offset for the dot-product of the input vectors for a polynomial or the bias offset for the dot-product of the input vectors used inside the hyperbolic tangent function for sigmoid kernels.  All other types ignore this value

The KernelParameters structure has only one other function, an initializer:

####init
<table border="1">
<tr>
	<td>Template</td>
	<td>init(type: SVMKernelType, degree: Int, gamma: Double, coef0: Double)</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method creates a KernelParameters with all fields initialized</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>(See member variable descriptions)</td>
</tr>
<tr>
	<td>Output</td>
	<td>N/A (constructor)</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

##LinearRegressionSubTerm
A LinearRegressionSubTerm structure is used to define a single item of a **LinearRegressionTerm**.  A subterm is represented mathematically as fn( i^p ), where i is the input value, p is the power value, and fn is the function modifier.  The subterms are multiplied together in the term, unless the subterm is marked as a divisor term.  The structure has the following member variables:

var | Type | Access | Description
--- | ---- | ------ | -------
	inputIndex | Int | get |  the index into the input vector for the value to be used in the subterm
	power | Double | get, set | the value to raise the input value to.  If exactly 1, no power function is performed
	function | SubtermFunction | get, set | the function that gets the input value raised to the power as an input, with the output being the result of the subterm value
	divide | Bool | get, set | if this boolean flag is true, the previous subterm values (that were multiplied together) gets divided by the value of this subterm, rather than the normal multiplication.  This flag is ignored on the first subterm

The LinearRegressionSubTerm structure has one constructor and one method:

####init
<table border="1">
<tr>
	<td>Template</td>
	<td>init(withInput: Int)</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method creates a LinearRegressionSubTerm with the specified input index.  The power value is set to 1, the function is set to '.none', and the divide flag is false</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>(See member variable description for inputIndex)</td>
</tr>
<tr>
	<td>Output</td>
	<td>N/A (constructor)</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

####getSubTermValue
<table border="1">
<tr>
	<td>Template</td>
	<td>getSubTermValue(_ withInputs: [Double]) throws -> Double</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method gets the value of the subterm, given the input vector</td>
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
		<td>withInputs</td>
		<td>[Double]</td>
		<td>the input vector</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>[Double] - The value for the subterm</td>
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
		<td>LinearRegressionError.negativeInLogOrPower</td>
		<td>thrown if power is less than zero and not negative, or a negative was passed to the logarithm function</td>
		</tr>
		</table>
	</td>
</tr>
</table>

##LinearRegressionTerm
The LinearRegressionTerm structure defines a single term in a linear regression model.  The model is of the form At<sub>1</sub> + Bt<sub>2</sub> + Ct<sub>3</sub>..., where A, B, and C, etc. are the parameter constants and t<sub>x</sub> are the terms.  Each term is a collection of **LinearRegressionSubTerm** elements that are multiplied (or divided) together.  The structure has two initializers and two public methods:

####init
<table border="1">
<tr>
	<td>Template</td>
	<td>init()</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method creates a LinearRegressionTerm with no LinearRegressionSubTerms.  All terms must be added manually</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>None</td>
</tr>
<tr>
	<td>Output</td>
	<td>N/A (constructor)</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

####initwithInput:atPower
<table border="1">
<tr>
	<td>Template</td>
	<td>init(withInput: Int, atPower: Double)</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method creates a LinearRegressionTerm with a single LinearRegressionSubTerm with the specified input index and power value.</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>(See <strong>LinearRegressionSubTerm</strong> description for parameters)</td>
</tr>
<tr>
	<td>Output</td>
	<td>N/A (constructor)</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

####addSubTerm
<table border="1">
<tr>
	<td>Template</td>
	<td>addSubTerm(_ subTerm: LinearRegressionSubTerm)</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method adds the subterm passed in to the end of the subterm list for the term</td>
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
		<td>subTerm</td>
		<td>LinearRegressionSubTerm</td>
		<td>subterm to add to the term</td>
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
	<td>No</td>
</tr>
</table>

####getTermValue
<table border="1">
<tr>
	<td>Template</td>
	<td>getTermValue(_ withInputs: [Double]) throws -> Double</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method gets the value of the term, given the input vector</td>
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
		<td>withInputs</td>
		<td>[Double]</td>
		<td>the input vector</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>[Double] - The value for the term</td>
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
		<td>LinearRegressionError.negativeInLogOrPower</td>
		<td>thrown if power is less than zero and not negative, or a negative was passed to the logarithm function</td>
		</tr>
		<tr>
		<td>LinearRegressionError.divideByZero</td>
		<td>thrown if a subterm with the division flag set to true equates to zero</td>
		</tr>
		</table>
	</td>
</tr>
</table>

##MDPEpisode 
An MDPEpisode structure defines an episode for a Markov Decision Process problem that has integer states and integer actions.  The structure holds the start state and all state transitions (with reward values) for the entire episode.  The structure has two member variables:

var | Type | Access | Description
--- | ---- | ------ | -------
	startState | Int | get, set |  the state index at the start of the episode
	events | [(action: Int, resultState: Int, reward: Double)] | get, set | array of transition tuples that comprise the episode.  Each tuple has the action taken from the previous state, the resulting state index, and the reward value received from transitioning to the new state
    
An MDPEpisode structure has a single initializer:

####init
<table border="1">
<tr>
	<td>Template</td>
	<td>init(startState: Int)</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method creates a MDPEpisode structure with the specified initial state, and an empty transition list</td>
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
		<td>startState</td>
		<td>Int</td>
		<td>the initial state index for the episode</td>
		</tr>
		</table>
</td>
</tr>
<tr>
	<td>Output</td>
	<td>N/A (constructor)</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

An MDPEpisode structure has a single public method:

####getTermValue
<table border="1">
<tr>
	<td>Template</td>
	<td>addEvent(_ event: (action: Int, resultState: Int, reward: Double))</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method adds the specified transition event to the end of the episode event list</td>
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
		<td>event</td>
		<td>(action: Int, resultState: Int, reward: Double)</td>
		<td>the event tuple containing the action taken, the resulting state, and the reward gained</td>
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


##PGStep 
An PGStep structure defines a single step of a policy-gradient reinforcement learning episode.  The step is managed by a **PGEpisode** class.  The structure has two member variables:

var | Type | Access | Description
--- | ---- | ------ | -------
	state | [Double] | get, set |  the state the beginning of this step of the episode
	gradient | [Double] | get, set |  the action gradient for this step of the episode.  This is the difference in the learning mechanism (neural network) output and the idealized action taken
	reward | Double | get, set | the reward for taking the action that makes up this step.

    
An PGStep structure has a single initializer:

####init
<table border="1">
<tr>
	<td>Template</td>
	<td>init(state: [Double], gradient: [Double], reward: Double)</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method creates a PGStep structure with the specified initial state, gradient, and reward</td>
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
		<td>state</td>
		<td>[Double]</td>
		<td>the initial state for the step</td>
		</tr>
		<tr>
		<td>gradient</td>
		<td>[Double]</td>
		<td>the initial action gradient for the step</td>
		</tr>
		<tr>
		<td>reward</td>
		<td>Double</td>
		<td>the reward the step</td>
		</tr>
		</table>
</td>
</tr>
<tr>
	<td>Output</td>
	<td>N/A (constructor)</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

