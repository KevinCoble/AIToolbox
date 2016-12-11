#Classes
The following public classes are available:

##AlphaBetaGraph
The AlphaBetaGraph class is used to the best current 'move' using alpha-beta graph pruning.  It uses nodes that conform to the **AlphaBetaNode** protocol.  The nodes generate the list of 'moves' that follow from the state position represented by that particular node.  This allows the graph to be generated dynamically, so no node representations are kept by the AlphaBetaGraph itself.  The AlphaBetaGraph can find the best current move using alpha-beta pruning linearly, or using concurrent threading for each 'move' below the current evaluation node for speed optimization.

The AlphaBetaGraph class has the an initializer and the following functions:
###init
<table border="1">
<tr>
	<td>Template</td>
	<td>init()</td>
</tr>
<tr>
	<td>Description</td>
	<td>Initializer for the AlphaBetaGraph class</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>None</td>
</tr>
<tr>
	<td>Output</td>
	<td>None (initializer)</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

###startAlphaBetaWithNode
<table border="1">
<tr>
	<td>Template</td>
	<td>startAlphaBetaWithNode(_ startNode: AlphaBetaNode, forDepth: Int, startingAsMaximizer : Bool = true) -> AlphaBetaNode?</td>
</tr>
<tr>
	<td>Description</td>
	<td>This performs an alpha-beta tree search starting at the node passed in, and going for the specified depth of moves.  Unless explicitly configured otherwise, the first move is considered as the move for the player maximizing the score.</td>
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
		<td>startNode</td>
		<td>AlphaBetaNode</td>
		<td>The node at the start of the search.  Child nodes will be generate dynamically by each node.</td>
		</tr>
		<tr>
		<td>forDepth</td>
		<td>Int</td>
		<td>The depth of the search.  The start node will generate children for depth 1.  Those children will generate grandchildren for depth 2, etc.</td>
		</tr>
		<tr>
		<td>startingAsMaximizer </td>
		<td>Bool</td>
		<td>Defaults to true.  If set to false, the moves generated from the starting node are assumed to nodes that should minimize the score, rather than maximize it.</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>AlphaBetaNode? - The immediate child node that represents the best move found.  If nil, no valid move was found.</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

###startAlphaBetaConcurrentWithNode
<table border="1">
<tr>
	<td>Template</td>
	<td>startAlphaBetaConcurrentWithNode(_ startNode: AlphaBetaNode, forDepth: Int, startingAsMaximizer : Bool = true) -> AlphaBetaNode?</td>
</tr>
<tr>
	<td>Description</td>
	<td>This performs an alpha-beta tree search starting at the node passed in, and going for the specified depth of moves.  Unless explicitly configured otherwise, the first move is considered as the move for the player maximizing the score.  The child score evaluation for each node will be carried out concurrently using Grand Central Dispatch</td>
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
		<td>startNode</td>
		<td>AlphaBetaNode</td>
		<td>The node at the start of the search.  Child nodes will be generate dynamically by each node.</td>
		</tr>
		<tr>
		<td>forDepth</td>
		<td>Int</td>
		<td>The depth of the search.  The start node will generate children for depth 1.  Those children will generate grandchildren for depthe 2, etc.</td>
		</tr>
		<tr>
		<td>startingAsMaximizer </td>
		<td>Bool</td>
		<td>Defaults to true.  If set to false, the moves generated from the starting node are assumed to nodes that should minimize the score, rather than maximize it.</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>AlphaBetaNode? - The immediate child node that represents the best move found.  If nil, no valid move was found.</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

##ClassificationData
The ClassificationData class is a class that has the class labels found in a classification data set, and the number of data points that belong to each class.  A protocol method called groupClasses in the **MLClassificationDataSet** protocol will get this information for a classification data set.  This class is often added as the optional data to a classification data set by model training methods.

##ConstraintProblem
The ConstraintProblem class is used to find solutions to a Constraint Propogation Problem.  The class should be set up with an array of **ConstraintProblemNode** objects, each with a constrained variable and set of constraints.  The problem can then be solved (if possible) using several different methods provided by the class.

##ConstraintProblemNode
The ConstraintProblemNode class represents a node in a Constraint Propogation Problem.  The nodes are 'connected' by a series of constraints that put limitations on the value of a variable.  The ConstraintProblemNode class has a **ConstraintProblemVariable** class to represent the constrained variable, and an array of classes that conform to the  **ConstraintProblemConstraint** protocol to represent the 'connections'.  A set of these nodes is passed to the **ConstraintProblem** class to find variable values for each node that satisfies the constrains, if such a solution exists.

The ConstraintProblemNode class has the following methods:

###init
<table border="1">
<tr>
	<td>Template</td>
	<td>init(variableDomainSize: Int)</td>
</tr>
<tr>
	<td>Description</td>
	<td>Initializer for the ConstraintProblemNode class.  The initialier creates a ConstraintProblemVariable for the node</td>
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
		<td>variableDomainSize</td>
		<td>Int</td>
		<td>The size of the domain for the variable for the node.</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>None (initializer)</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

###resetVariable
<table border="1">
<tr>
	<td>Template</td>
	<td>resetVariable()</td>
</tr>
<tr>
	<td>Description</td>
	<td>Resets the node's variable to not being assigned and removes all restrictions on the domain to what the variable can be assigned to</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>None</td>
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

###processSelfConstraints
<table border="1">
<tr>
	<td>Template</td>
	<td>processSelfConstraints(_ graphNodeList: [ConstraintProblemNode]) -> Bool</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method iterates through all of the constraints assigned to the node and has each one that is a constraint on this node enforce itself on the graph that is passed in.  It returns a flag indicating if the node variable still has possible assignments after the constrain enforcement</td>
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
		<td>the graph (collection of nodes) that this node is a part of.  Constraint node indexes are indexes into the array.</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>Bool - flag indicating if the node variable still has possible assignments after the constrain enforcement</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

###clearConstraintsLastEnforced
<table border="1">
<tr>
	<td>Template</td>
	<td>clearConstraintsLastEnforced()</td>
</tr>
<tr>
	<td>Description</td>
	<td>Removes the list of constraints enforced by the last call to processSelfConstraints.</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>None</td>
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

###enforceConstraints
<table border="1">
<tr>
	<td>Template</td>
	<td>enforceConstraints(_ graphNodeList: [ConstraintProblemNode], nodeEnforcingConstraints: ConstraintProblemNode) -> Bool</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method iterates through all of the constraints assigned to the enforcing node and has each one enforce itself on the graph that is passed in.  The enforced constraints are saved in a list (cleared by a clearConstraintsLastEnforced call), so the effects can be undone for search backup.  The method always returns true</td>
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
		<td>the graph (collection of nodes) that this node is a part of.  Constraint node indexes are indexes into the array.</td>
		</tr>
		<tr>
		<td>nodeEnforcingConstraints</td>
		<td>ConstraintProblemNode</td>
		<td>the node that is the source of the constraints being enforced.</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>Bool - flag indicating if the node variable still has possible assignments after the constrain enforcement</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

###removeConstraintsLastEnforced
<table border="1">
<tr>
	<td>Template</td>
	<td>removeConstraintsLastEnforced()</td>
</tr>
<tr>
	<td>Description</td>
	<td>Removes the constraints enforced by the last call to processSelfConstraints.  Used for backing-up the graph search</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>None</td>
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

###resetVariableDomainIndex
<table border="1">
<tr>
	<td>Template</td>
	<td>resetVariableDomainIndex(_ resetIndex: Int)</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method resets any constraint on the attached variable associated with a particular domain index.  This will allow the variable to be able to be assigned to that index later.</td>
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
		<td>resetIndex</td>
		<td>Int</td>
		<td>The domain index to have reset (made available) on the internal variable.</td>
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

###assignSingleton
<table border="1">
<tr>
	<td>Template</td>
	<td>assignSingleton() ->Bool</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method determines if the associated variable is a singleton - having only one remaining possible value.  If not it returns false.  If it is a singleton, the variable is assigned the value that remains as a possibility.</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>None</td>
</tr>
<tr>
	<td>Output</td>
	<td>Bool - flag indicating if the associtated variable is a singleton</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

##ConstraintProblemVariable
The ConstraintProblemVariable class is used to represent a constrained variable in a Constraint Propogation Problem.  An example of a constrained variable is the color in the three-color map problem.  The color of a node is constrained by adjacent nodes.  In this problem the variable is the color, and the domain size would be three (three possible colors).  A **ConstraintProblemNode** object gets a constrained variable and a set of constraints (in the above example, the constraints would be 'not the same as node x', where x is an adjacent node).

The ConstraintProblemVariable class has six public variables/properties:

var | Type | Access | Description
--- | ---- | ------ | -------
domainSize| Int | get | the number of possible states in the domain for this variable
hasNoPossibleSettings | Bool | get | flag indicating there are no remaining open states this variable can be assigned to
isSingleton | Bool | get | flag indicating this variable has only one remaining state it can be assigned to
assignedValue | Int? | get, set | the current state assignment for this variable.  If nil, the variable is unassigned
smallestAllowedValue | Int? | get | the lowest numbered state still available for assignment to this variable.  Returns nil if no possible states remain for the variable
largestAllowedValue | Int? | get | the highest numbered state still available for assignment to this variable.  Returns nil if no possible states remain for the variable

One initializer and five public methods are available

###init
<table border="1">
<tr>
	<td>Template</td>
	<td>init(sizeOfDomain: Int)</td>
</tr>
<tr>
	<td>Description</td>
	<td>Initializer for the ConstraintProblemVariable class</td>
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
		<td>sizeOfDomain</td>
		<td>Int</td>
		<td>The size of the domain of the variable, which is the number of unique states the variable can have assigned to it.</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>None (initializer)</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

###reset
<table border="1">
<tr>
	<td>Template</td>
	<td>reset()</td>
</tr>
<tr>
	<td>Description</td>
	<td>Resets the variable to not being assigned and removes all restrictions on the domain to what the variable can be assigned to</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>None</td>
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

###removeValuePossibility
<table border="1">
<tr>
	<td>Template</td>
	<td>removeValuePossibility(_ varValueIndex: Int) -> Bool</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method removes the specified domain instance from the possible assignments for this variable</td>
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
		<td>varValueIndex</td>
		<td>Int</td>
		<td>index of domain value that will no longer be able to be assigned to this variable</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>Bool - flag indicating if the domain index was possible to be assigned before the method was invoked</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

###allowValuePossibility
<table border="1">
<tr>
	<td>Template</td>
	<td>allowValuePossibility(_ varValueIndex: Int) -> Bool</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method adds the specified domain instance to the possible assignments for this variable</td>
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
		<td>varValueIndex</td>
		<td>Int</td>
		<td>index of domain value that will now be able to be assigned to this variable</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>Bool - flag indicating if the domain index was added as a possibility, rather than it already being one</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

###assignToNextPossibleValue
<table border="1">
<tr>
	<td>Template</td>
	<td>assignToNextPossibleValue() ->Bool</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method assigns the variable to the first possible value in the domain if the variable is currenly unasigned, else it changes the assignment to the next value possible for the variable in the domain.  If no other possible (higher) value remains for the variable, the method returns false</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>None</td>
</tr>
<tr>
	<td>Output</td>
	<td>Bool - flag indicating if the variable has been assigned a new value</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

###assignSingleton
<table border="1">
<tr>
	<td>Template</td>
	<td>assignSingleton() ->Bool</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method determines if the variable is a singleton - having only one remaining possible value.  If not it returns false.  If it is a singleton, the variable is assigned the value that remains as a possibility.</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>None</td>
</tr>
<tr>
	<td>Output</td>
	<td>Bool - flag indicating if the variable is a singleton</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

##Convolution2D : DeepNetworkOperator
The Convolution2D class is a **DeepNetworkOperator** that convolves the incoming two-dimensional matrix using a convolution matrix that can be either fixed or learned.  The result is a two-dimensional matrix of the same size as the input.

##DataSet : MLRegressionDataSet, MLClassificationDataSet, MLCombinedDataSet
The DataSet class is a generic data source class that conforms to all of the data protocols used by the AIToolbox framework, **MLRegressionDataSet**, **MLClassificationDataSet**, and **MLCombinedDataSet**.  This allows the class to be the generic workhorse for all data set needs in machine learning.  If you have a custom class that obtains your data, have that class conform to the appropriate protocol.  Otherwise, use the DataSet class to contain the data you get from sources you do not have the ability to easily modify to conform to the necessary protocol(s).

##DeepChannel : MLPersistence
The DeepChannel class represents a 'channel' within a layer, which is itself part of a deep-network.  Each layer has one or more channels to process data from the previous layer, or from the network inputs if in the first layer.  The channels are independent, and can be processed concurrently.  The channel contains a series of operators (**DeepNetworkOperator** conforming classes) that process the specified labeled data from the previous layer into output data.  The channel then labels the output data for reference by the succeeding layers' channels.

##DeepLayer : DeepNetworkInputSource, DeepNetworkOutputDestination, MLPersistence
The DeepLayer class represents a 'layer' within a deep-network.  Each layer has one or more channels to process data from the previous layer, or from the network inputs if it is the first layer.  The final layer's output is the result of the deep-network.

##DeepNetwork : DeepNetworkInputSource, DeepNetworkOutputDestination, MLPersistence
The DeepNetwork class is the top-level class for a deep-network.  The class manages arrays of layers (**DeepLayer** objects) and input data sets (**DeepNetworkInput** objects).  Adding, modifying, and removing operators, channels, and layers can be done at the DeepNetwork level.  Once the network is configured, input data can be set on the class and the resulting value(s) calculated.  Unlike the **NeuralNetwork** class, deep-networks use single-precision mathematics.

##DeepNeuralNetwork : DeepNetworkOperator
The DeepNeuralNetwork class is a **DeepNetworkOperator**, an operation done within a channel of a deep-network.  The DeepNeuralNetwork operation is a standard learnable feed-forward neural network layer.  The number of neurons in the neural layer (not to be confused with a DeepLayer) is given by the size structure used to create the operation.  There will be one node for each output in the size structure.  The input data will be fully connected to each of the nodes.

##DoubleGene
The DoubleGene is used as a floating-value gene within a **Genome**.  The floating values are single-precision values with a configurable range.  Mating and mutation is performed at a value level within each allele, with all results clipped to the valid range for the gene.

##Gaussian
The Gaussian class represents a single-dimensional gaussian probability distribution, with a given mean and variance.  Once initialized, the class can be queried to get a probability of an input value, or a random value that comes from the distribution.  The class also contains the static function for getting a generic gaussian-distributed random value.

##Genome
The Genome class represents the genetic material of a single member of a genetic population managed by a **Population** object.  Each genome consists of a set of integer and/or double-valued genes (**IntegerGene** and **DoubleGene** classes), each of which have a configurable number of alleles.

##Graph&lt;T : GraphNode&gt;
The Graph class represents a directed graph of nodes connected by links.  The Graph class uses a generic **GraphNode** template for managing the nodes, which are linked by **GraphEdge** objects or appropriate subclassed objects.  The Graph class is used to search for a path from a starting node to a node matching a specified search criteria.

##GraphEdge
The GraphEdge class defines an edge between two nodes in a directed graph.  The source node owns the GraphEdge, while the destination node is specified as a member variable of this class.  The class has a single Double value for the 'cost' value of the edge.  Override the class if the cost is variable and needs to be calculated dynamically.

##GraphNode
The GraphNode class defines a single node in a directed graph.  The nodes are connected by **GraphEdge** objects (or derived subclass objects) that GraphNode manages if it is the source node in the link.  The class should be sub-classed if data is needed on each node (such as node meaning or data regarding link cost calculations).

##IntegerGene
The IntegerGene is used as an integer gene within a **Genome**.  The integer values are signed 32-bit values.  Mating and mutation is performed at a bit-wise level within each allele.

##InternalConstraint: ConstraintProblemConstraint
The InternalConstraint class is a concrete class that implements the **ConstraintProblemConstraint** protocol.  The class represents a constriant in a Constraint Propogation Problem.  For most problems that have discrete variables, this constraint class will be all that is needed for the constraints, if the type of the constraint can come from the **StandardConstraintType** enumeration.

Besides the variable and methods that come from the **ConstraintProblemConstraint** protocol, the InternalConstraint class has the following methods:

###init
<table border="1">
<tr>
	<td>Template</td>
	<td>init(type: StandardConstraintType, index: Int)</td>
</tr>
<tr>
	<td>Description</td>
	<td>Initializer for the InternalConstraint class</td>
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
		<td>type</td>
		<td>StandardConstraintType</td>
		<td>The type of the constraint on the nodes' variable.</td>
		</tr>
		<tr>
		<td>index</td>
		<td>Int</td>
		<td>If the constraint type is "can't be value", this index is the value the variable cannot be.  For constraints that reference another node (for example, the "can't be same value as other node" type), this index is the node index within the graph for the referenced node.</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>None (initializer)</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

###reciprocalConstraint
<table border="1">
<tr>
	<td>Template</td>
	<td>reciprocalConstraint(_ firstNodeIndex: Int) ->InternalConstraint?</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method returns a InternalConstrain class object, initialized to have the reciprocal constraint from the one indicated by the instance the method was called on.  This allows the setup code to set opposite node with the reverse constraint</td>
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
		<td>firstNodeIndex</td>
		<td>Int</td>
		<td>index of node that the reverse constraint should be created from (usually the index for the node instance that the method is called on).  This will be passed as the target node when creating the new constraint</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>InternalConstraint? - the constraint created, if it was possible to create the reciprocol.  Constraints of the "can't be value" type will always return nil</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>


##KMeans
The KMeans class is used to take an unlabelled classification data set (from a class that conforms to the **MLClassificationDataSet** protocol) and finds a set of labels for the data, grouping data that is close together in state space by giving them the same label.

##LinearRegressionModel : Regressor
The LinearRegressionModel class is used to find a linear model that best matches a given data set using regression techniques.  The model is linear in parameter space, not necessarily in input space.  Therefore functions like Ax + Bx<sup>2</sup> + Ce<sup>x</sup> can be used, as each parameter (A, B, and C) are used linearly, even if the input (a single dimension vector x in this case) is not used linearly.  The model equation is represented by a series of **LinearRegressionTerm** structures.  A convenience constructor is available for simple polynomial models, as they are very common.  Model selection can be done using LinearRegressionModel (and other regressors) in a **Validation** class.

##LogisticRegression : Regressor, Classifier
The LogisticRegression class can be used as either a regressor or a classifier.  The class models the regression values or the discrimination line using a logistic function (1 / 1+e<sup>-sum</sup>, where sum is the dot product of the input vector with the parameter vector).  As the class conforms to both the **Regressor** and **Classifier** protocols, the class can be used in a **Validation** class for model selection.

##MDP
The MDP class is used to generate a policy (a recommended action for each state) of a Markov Decision Process problem.  The action list must be discrete, but the state space can be discrete or continuous, depending on the solution method to be used.  Most of the solution functions take functions as parameters, where these functions generate the action lists, the result state of taking an action, or the reward (or punishment) received for the action result.  The results is either a list of optimal actions for each state (if the state space is discrete), or a parameter set that can be used to get the optimal action for a continuous state vector.

##MetalNeuralNetwork
The MetalNeuralNetwork class uses Apple's Metal API to train a feed-forward neural network using the graphics processor.

##MixtureOfGaussians : Regressor
The MixtureOfGaussians class is a regression model that uses a weighted average of multi-variate gaussian distributions to compute the regression value.  The the class conforms to the **Regressor** protocol.

##MLLegendItem
The MLLegendItem class represents an object added to an **MLViewLegend** object.  A legend item has two parts, the label and the symbol or line.  The legend items are drawn in the order they are added to the legend, with the graphic drawn centered to the right of the label.  Constructors are provided that can take a **ViewRegressionDataSet**, **MLViewRegressionLine**, or **MLViewClassificationDataSet** and extract the pertinent information for the legend graphic.

##MLPlotSymbol
The MLPlotSymbol class is used to plot a symbol on a data point within an **MLView** plot.  The class is used in data sets plot objects and legend labels.  The base class provides various shapes that can be sized and colored as desired.  A sub-class can be used to add additional custom shapes.

##MLView: NSView
The MLView class is a NSView or UIView (depending on if the framework was built for macOS or iOS) derived view that can be used to show a variety of machine-learning data types, such as regression or classification data sets, regression result curves, and/or classification decision areas.  Any object that conforms to the **MLViewItem** protocol can be added to the view.  Labels and legends can be added to identify items being plotted.  Items are drawn in the order they are added to the MLView object.

##MLViewAxisLabel: MLViewItem
The MLViewAxisLabel class draws X and/or Y axis labels onto an **MLView** plot.  Besides being able to turn on or off either axis label, the number of major and minor tick divisions, the size of the tick marks, the font used for the labels, and the label number formatting can be specified.  If labels are turned on, the data plot area is reduced to make room for the labels, as they are drawn outside the data area.

##MLViewClassificationArea: MLViewItem 
The MLViewClassificationArea class draws a colored area showing the class label associated areas (from a class that conforms to the **Classifier** protocol) on an **MLView** plot.  The class uses the specified X and Y axis source (input vector member) to determine a class label for the center of a rectangle within the plot area, where the rectangle will be drawn in a specified color for that label.  The number of rectangles the plot area is separated into is specified with a granularity adjustment.

##MLViewClassificationDataSet: MLViewItem
The MLViewClassificationDataSet class draws a symbol on every data point of a classification data set (a data class that conforms to the **MLClassificationDataSet** protocol) within an **MLView** plot.  The symbol used for each class label in the data set can be configured.  A MLViewClassificationDataSet can provide the scale for the plot, using the range of data values in the data set as the scaling parameters.

##MLViewLegend: MLViewItem
The MLViewLegend class draws a plot legend on an **MLView** plot at a specified location.  The items in the legend consist of an array of **MLLegendItem** classes.  The size of the legend will be dynamically determined from the item count, label strings, and font selection used.

##MLViewRegressionDataSet: MLViewItem
The MLViewRegressionDataSet class draws a configurable symbol on every data point of a regression data set (a data class that conforms to the **MLRegressionDataSet** protocol) within an **MLView** plot.  A MLViewRegressionDataSet can provide the scale for the plot, using the range of data values in the data set as the scaling parameters.

##MLViewRegressionLine: MLViewItem
The MLViewRegressionLine class draws a line following a regression model (from a class that conforms to the **Regressor** protocol) on an **MLView** plot, using the specified X axis source (input vector member, output vector member, or class label) to calculate a Y axis position for every point across the plot X axis.  The line is plotted using a specified color and line thickness.

##MultivariateGaussian
The MultivariateGaussian class represents a multi-dimensional gaussian probability distribution, with a given mean vector and a variance matrix.  The distribution can be configured as 'diagonal', meaning all non-diagonal terms in the variance matrix are assumed to be zero.  Once initialized, the class can be queried to get a probability of an input vector, or a random vector that comes from the distribution.

##NeuralNetwork: Classifier, Regressor
The NeuralNetwork class manages a series of internal network layers that comprise a neural network.  The class provides initialization methods to define these layers, which can be of various types such as feed-forward or recurrent layers.  Once a network is configured methods are provided that can train the network parameters in various ways.  Once trained, the network can be used to get results for new input vectors.  The NeuralNetwork class conforms to both the **Classifier** and **Regressor** protocols, so it can be used in both classification and regression roles.

##NonLinearRegression : Regressor
The NonLinearRegression class is is used to find a non-linear model that best matches a given data set using regression techniques.  The model can be any equation that can be parameterized, as a class that conforms to the **NonLinearEquation** protocol is passed in to represent the equation.  Various methods can be used to attempt to solve the regression model, like gradient-descent and gauss-newton.  Model selection can be done using LinearRegressionModel (and other regressors) in a **Validation** class.

##PCA
The PCA class preforms 'Principal Component Analysis', which is used to analyze a data set to find a set of basis vectors that still represent the data well, but have a lower dimension.  This will often make large problem train better in other models like neural networks.  The algorithm finds the eigenvalues and eigenvectors of the data matrix, one for each element of the input vector.  The eigenvalues give the relative 'importance' of that element of the input vector, so the smaller values can be removed.  A method is provided that will map an input vector onto the eigenvector basis from the remaining set, giving the smaller dimension vector that can be used in further data analysis.

##Population
The Population class represents a population of objects that have genetic characteristics - in this case an array of **Genome** classes.  The Population class is then used to 'breed' the next generation using sexual or asexual reproduction with a specified mutation rate.  The individuals used to create the new members are selected randomly from the current population, but weighted towards individuals with a higher 'score' value.  The score value is a function of the individual members of the population, and should be set before creating the next generation.

##Pooling : DeepNetworkOperator
The Pooling class is a **DeepNetworkOperator** that reduces the incoming data matrix by a specified amount in each dimension using an average, maximum, or minimum operation.  The result is a reduced-size matrix of the same number of dimensions as the input.

##Queue&lt;T&gt;
The Queue class is a generic class that manages a first-in-first-out queue of the data type.  Several graph search strategies use queues to maintain a list of nodes to be processed.

##SVMModel : Classifier
The SVMModel class represents a support vector machine.  Support vector machines are models, generally used for classification - hence the conformance to the **Classifier** protocol, that mathematically determine the decision boundaries between classes by maximizing the  margin between the boundary and the training data points.  The class is a direct port of the LIBSVM code by Chih-Chung Chang and Chih-Jen Lin, with additions made to support data set classes and protocols from the AIToolbox framework.

##Validation
The Validation class is configured with a set of regression (conforming to the **Regressor** protocol) or classifier (conforming to the **Classifiier** protocol) models, and uses a training/testing dataset to find the model that performs the best given the model parameters and the training data.  This can be used to select from different model types (as long as they are all regressors or classifiers), or to find configuration parameters that are close to optimal for the problem.


