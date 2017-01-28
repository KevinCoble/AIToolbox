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
The ConstraintProblem class has the following methods:

###init
<table border="1">
<tr>
	<td>Template</td>
	<td>init()</td>
</tr>
<tr>
	<td>Description</td>
	<td>Initializer for the ConstraintProblem class.  The initialier creates an empty ConstraintProblem</td>
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

###setNodeList
<table border="1">
<tr>
	<td>Template</td>
	<td>setNodeList(_ list: [ConstraintProblemNode])</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method sets the nodes for the problem.  Each node will be renumbered based on it's order in the array passed in.</td>
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
		<td>list</td>
		<td>[ConstraintProblemNode]</td>
		<td>The nodes that are to make up the constraint problem graph.</td>
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

###clearConstraints
<table border="1">
<tr>
	<td>Template</td>
	<td>clearConstraints()</td>
</tr>
<tr>
	<td>Description</td>
	<td>Initializer clears all the constraints from all of the nodes in the graph</td>
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

###addValueConstraintToNode
<table border="1">
<tr>
	<td>Template</td>
	<td>addValueConstraintToNode(_ node: Int, invalidValue: Int)</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method adds a value constraint to a specified node.  This constraint will make the node not able to have its variable assigned to the value indicated.</td>
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
		<td>node</td>
		<td>Int</td>
		<td>The index of the node within the graph to get this constraint.</td>
		</tr>
		<tr>
		<td>invalidValue</td>
		<td>Int</td>
		<td>The index of the domain value for the variable that will no longer be allowed on the node.</td>
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

###addConstraintOfType
<table border="1">
<tr>
	<td>Template</td>
	<td>addConstraintOfType(_ type: StandardConstraintType,  betweenNodeIndex firstnode: Int, andNodeIndex secondNode : Int)</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method adds a constraint between two nodes.  The constraint is placed on the first node index specified, with the constraint having the type indicated, and referencing the second node.</td>
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
		<td>The type of constraint being added.</td>
		</tr>
		<tr>
		<td>betweenNodeIndex </td>
		<td>Int</td>
		<td>The index of the node within the graph to get this constraint.</td>
		</tr>
		<tr>
		<td>andNodeIndex </td>
		<td>Int</td>
		<td>The index of the node that will be referenced by the constraint, if the constraint type requires a referencing node.</td>
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

###addReciprocalConstraintsOfType
<table border="1">
<tr>
	<td>Template</td>
	<td>addReciprocalConstraintsOfType(_ type: StandardConstraintType,  betweenNodeIndex firstnode: Int, andNodeIndex secondNode : Int)</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method adds a set of two constraints between two nodes.  The constraint is placed on the first node index specified, with the constraint having the type indicated, and referencing the second node, while a reciprocal restraint is added to the second node, with the first node being referenced</td>
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
		<td>The type of constraint being added.  The inverse type will be added to the second node.</td>
		</tr>
		<tr>
		<td>betweenNodeIndex </td>
		<td>Int</td>
		<td>The index of the node within the graph to get the main constraint, while the second node gets the reciprocal.</td>
		</tr>
		<tr>
		<td>andNodeIndex </td>
		<td>Int</td>
		<td>The index of the node that will be referenced by the main constraint, and get the reciprocal constraint.</td>
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

###addCustomConstraint
<table border="1">
<tr>
	<td>Template</td>
	<td>addCustomConstraint(_ constraint: ConstraintProblemConstraint, toNode: Int)</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method adds a custom constraint to the node indicated.  The constraint can be an instance of **InternalConstraint**, or any custom class that conforms to the **ConstraintProblemConstraint** protocol</td>
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
		<td>constraint</td>
		<td>ConstraintProblemConstraint</td>
		<td>The constraint to be added added.  This is an instantiated class object.</td>
		</tr>
		<tr>
		<td>toNode</td>
		<td>Int</td>
		<td>The index of the node within the graph to get the custom constraint.</td>
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

###solveWithForwardPropogation
<table border="1">
<tr>
	<td>Template</td>
	<td>solveWithForwardPropogation() -> Bool</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method resets the variable possibilities for each node, processes self-inflicted constraints, then runs a depth-first-search - processing additional constraints as variables get assigned to nodes.  The method returns a flag indicating if a valid solution was found.  If a solution was found, each node in the graph will have its variable assigned to the value that makes that solution possible.</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>None</td>
</tr>
<tr>
	<td>Output</td>
	<td>Bool - flag indicating if a solution was found.</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

###solveWithSingletonPropogation
<table border="1">
<tr>
	<td>Template</td>
	<td>solveWithSingletonPropogation() -> Bool</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method resets the variable possibilities for each node, processes self-inflicted constraints, then runs a depth-first-search - processing additional constraints as variables get assigned to nodes.  If a node is left with a single possible value after these constraints are enforced, that value is assigned then (instead of waiting for the depth-first search to assign it) - and all constraint effects stemming from that assignment are enforced.  The method returns a flag indicating if a valid solution was found.  If a solution was found, each node in the graph will have its variable assigned to the value that makes that solution possible.</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>None</td>
</tr>
<tr>
	<td>Output</td>
	<td>Bool - flag indicating if a solution was found.</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

###solveWithFullPropogation
<table border="1">
<tr>
	<td>Template</td>
	<td>solveWithFullPropogation() -> Bool</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method resets the variable possibilities for each node, processes self-inflicted constraints, then runs a depth-first-search - processing additional constraints as variables get assigned to nodes.  After a constraint is enforced on a node, all constraints on that node are checked to see if the constraint affect propogates, and nodes further down the chain are processed if that is the case.  This takes additional calculation time, so may only be warrented if there are 'greater than' or 'less than' type constraints.  If a solution was found, each node in the graph will have its variable assigned to the value that makes that solution possible.</td>
</tr>
<tr>
	<td>Inputs</td>
	<td>None</td>
</tr>
<tr>
	<td>Output</td>
	<td>Bool - flag indicating if a solution was found.</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

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
Besides the methods required by the DeepNetworkOperator protocol (and the MLPersistence protocol it requires), the following methods are available for the class:

###init
<table border="1">
<tr>
	<td>Template</td>
	<td>init(usingMatrix : Convolution2DMatrix)<\td>
</tr>
<tr>
	<td>Description</td>
	<td>This initializer takes a matrix for the convolution and creates a Convolution2D operator object</td>
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
		<td>usingMatrix </td>
		<td>Convolution2DMatrix</td>
		<td>the matrix for the convolution performed by the operator</td>
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

###setMatrixType
<table border="1">
<tr>
	<td>Template</td>
	<td>setMatrixType(type : Convolution2DMatrix)<\td>
</tr>
<tr>
	<td>Description</td>
	<td>Sets the matrix (type and values) from the passed in structure into the class</td>
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
		<td>type </td>
		<td>Convolution2DMatrix</td>
		<td>the matrix for the convolution performed by the operator</td>
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

###setMatrixValue
<table border="1">
<tr>
	<td>Template</td>
	<td>setMatrixValue(atIndex: Int, toValue: Float)<\td>
</tr>
<tr>
	<td>Description</td>
	<td>Sets the value of the class convolution matrix at a specific index in the matrix</td>
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
		<td>atIndex</td>
		<td>Int</td>
		<td>the index in the matrix to be modified</td>
		</tr>
		</tr>
		<td>toValue</td>
		<td>Int</td>
		<td>the new value for the matrix element</td>
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

###determineResultRange
<table border="1">
<tr>
	<td>Template</td>
	<td>determineResultRange()<\td>
</tr>
<tr>
	<td>Description</td>
	<td>Sets the internal values of the minimum and maximum expected result values, based on the current convolution matrix</td>
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

##DataSet : MLRegressionDataSet, MLClassificationDataSet, MLCombinedDataSet
The DataSet class is a generic data source class that conforms to all of the data protocols used by the AIToolbox framework, **MLRegressionDataSet**, **MLClassificationDataSet**, and **MLCombinedDataSet**.  This allows the class to be the generic workhorse for all data set needs in machine learning.  If you have a custom class that obtains your data, have that class conform to the appropriate protocol.  Otherwise, use the DataSet class to contain the data you get from sources you do not have the ability to easily modify to conform to the necessary protocol(s).

The DataSet class has the member variables required by the **MLDataSet** protocol.  Much  of the functionality of the class comes from methods required by the **MLRegressionDataSet**, **MLClassificationDataSet**, and **MLCombinedDataSet** protocols.  See those protocols for details on these methods.  The following methods are initializers and additional functionality provide by the DataSet class:

### init
<table border="1">
<tr>
	<td>Template</td>
	<td>init(dataType : DataSetType, inputDimension : Int, outputDimension : Int)</td>
</tr>
<tr>
	<td>Description</td>
	<td>This initializer creates a DataSet object with specified type, and input and output dimensions</td>
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
		<td> dataType </td>
		<td> DataSetType </td>
		<td>the type of data that will be managed by this data set</td>
		</tr>
		<tr>
		<td> inputDimension </td>
		<td> Int </td>
		<td>the size of the input vector that will be in this data set</td>
		</tr>
		<tr>
		<td> outputDimension </td>
		<td> Int </td>
		<td>the size of the output vector that will be in this data set.  If the type is ‘.Classification’, this value will be ignored.</td>
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

<table border="1">
<tr>
	<td>Template</td>
	<td>init(fromDataSet: DataSet)</td>
</tr>
<tr>
	<td>Description</td>
	<td>This initializer creates a DataSet object that is a copy of the data set passed int</td>
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
		<td>fromDataSet</td>
		<td>DataSet</td>
		<td>the DataSet object to copy</td>
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

<table border="1">
<tr>
	<td>Template</td>
	<td>init?(fromRegressionDataSet: MLRegressionDataSet)</td>
</tr>
<tr>
	<td>Description</td>
	<td>This initializer creates a DataSet object that is a copy of the regression data set passed int</td>
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
		<td>fromRegressionDataSet</td>
		<td>MLRegressionDataSet</td>
		<td>the object that conforms to the MLRegressionDataSet protocol to copy</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>nil if data set passed in is not appropriate for copying</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

<table border="1">
<tr>
	<td>Template</td>
	<td>init?(fromClassificationDataSet: MLClassificationDataSet)</td>
</tr>
<tr>
	<td>Description</td>
	<td>This initializer creates a DataSet object that is a copy of the classification data set passed int</td>
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
		<td>fromClassificationDataSet</td>
		<td>MLClassificationDataSet</td>
		<td>the object that conforms to the MLClassificationDataSet protocol to copy</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>nil if data set passed in is not appropriate for copying</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

<table border="1">
<tr>
	<td>Template</td>
	<td>init?(fromCombinedDataSet: MLCombinedDataSet)</td>
</tr>
<tr>
	<td>Description</td>
	<td>This initializer creates a DataSet object that is a copy of the combined data set passed int</td>
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
		<td>fromCombinedDataSet</td>
		<td>MLCombinedDataSet</td>
		<td>the object that conforms to the MLCombinedDataSet protocol to copy</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>nil if data set passed in is not appropriate for copying</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

<table border="1">
<tr>
	<td>Template</td>
	<td>init?(dataType : DataSetType, withInputsFrom: MLDataSet)</td>
</tr>
<tr>
	<td>Description</td>
	<td>This initializer creates a DataSet object that is of the specified type, but with the input arrays set from the passed in data set</td>
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
		<td>dataType </td>
		<td>DataSetType</td>
		<td>the type of data set to create</td>
		</tr>
		<tr>
		<td>withInputsFrom</td>
		<td>MLDataSet</td>
		<td>the data set object to copy the input vectors from</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>nil if data set passed in is not appropriate for copying</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

<table border="1">
<tr>
	<td>Template</td>
	<td>init?(fromDataSet: MLDataSet, withEntries: [Int])</td>
</tr>
<tr>
	<td>Description</td>
	<td>This initializer creates a DataSet object that is a copy of the data set object passed in, but with a subset of the data points</td>
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
		<td>fromDataSet</td>
		<td>MLDataSet</td>
		<td>the data set object to configure this data set from and to copy a subset if data out of</td>
		</tr>
		<tr>
		<td>withEntries</td>
		<td>[Int]</td>
		<td>an array of indices into the data set to copy into this object</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>nil if data set passed in is not appropriate for copying or any of the indices are out of range</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

<table border="1">
<tr>
	<td>Template</td>
	<td>init?(fromDataSet: MLDataSet, withEntries: ArraySlice&lt;Int&gt;)</td>
</tr>
<tr>
	<td>Description</td>
	<td>This initializer creates a DataSet object that is a copy of the data set object passed in, but with a subset of the data points</td>
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
		<td>fromDataSet</td>
		<td>MLDataSet</td>
		<td>the data set object to configure this data set from and to copy a subset if data out of</td>
		</tr>
		<tr>
		<td>withEntries</td>
		<td>ArraySlice&lt;Int&gt;</td>
		<td>a slice of an array of indices into the data set to copy into this object</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>nil if data set passed in is not appropriate for copying or any of the indices are out of range</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

###includeEntries
<table border="1">
<tr>
	<td>Template</td>
	<td>includeEntries(fromDataSet: MLDataSet, withEntries: [Int]) throws</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method adds the entries from the specified data set that match the array of indices provide to this data set</td>
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
		<td>fromDataSet</td>
		<td>MLDataSet</td>
		<td>the data set to get the new entries from</td>
		</tr>
		<tr>
		<td>withEntries</td>
		<td>[Int]</td>
		<td>an array of indices for data points to copy into this data set</td>
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
		<td>DataTypeError.invalidDataType</td>
		<td>thrown if the data set passed in is not a type that works with the data set the method is called on</td>
		</tr>
		<tr>
		<td>DataTypeError.wrongDimensionOnInput</td>
		<td>thrown if the input dimension for the data set passed in does not match the data set the method is called on</td>
		</tr>
		<tr>
		<td>DataTypeError.wrongDimensionOnOutput</td>
		<td>thrown if the output dimension for the data set passed in does not match the data set the method is called on</td>
		</tr>
		<tr>
		<td>DataIndexError.negative</td>
		<td>thrown if an index in the passed in array is negative</td>
		</tr>
		<tr>
		<td>DataIndexError.indexAboveDataSetSize</td>
		<td>thrown if an index in the passed in array is above the last data point index of the passed in data set</td>
		</tr>
		</table>
	</td>
</tr>
</table>

<table border="1">
<tr>
	<td>Template</td>
	<td>includeEntries(fromDataSet: MLDataSet, withEntries: ArraySlice&lt;Int&gt;) throws</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method adds the entries from the specified data set that match the array of indices provide to this data set</td>
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
		<td>fromDataSet</td>
		<td>MLDataSet</td>
		<td>the data set to get the new entries from</td>
		</tr>
		<tr>
		<td>withEntries</td>
		<td>ArraySlice&lt;Int&gt;</td>
		<td>an array slice of indices for data points to copy into this data set</td>
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
		<td>DataTypeError.invalidDataType</td>
		<td>thrown if the data set passed in is not a type that works with the data set the method is called on</td>
		</tr>
		<tr>
		<td>DataTypeError.wrongDimensionOnInput</td>
		<td>thrown if the input dimension for the data set passed in does not match the data set the method is called on</td>
		</tr>
		<tr>
		<td>DataTypeError.wrongDimensionOnOutput</td>
		<td>thrown if the output dimension for the data set passed in does not match the data set the method is called on</td>
		</tr>
		<tr>
		<td>DataIndexError.negative</td>
		<td>thrown if an index in the passed in array slice is negative</td>
		</tr>
		<tr>
		<td>DataIndexError.indexAboveDataSetSize</td>
		<td>thrown if an index in the passed in array slice is above the last data point index of the passed in data set</td>
		</tr>
		</table>
	</td>
</tr>
</table>

###includeEntryInputs
<table border="1">
<tr>
	<td>Template</td>
	<td>includeEntryInputs(fromDataSet: MLDataSet, withEntries: [Int]) throws</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method adds the entry inputs from the specified data set that match the array of indices provide to this data set.  Outputs ar set to zero</td>
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
		<td>fromDataSet</td>
		<td>MLDataSet</td>
		<td>the data set to get the new entries from</td>
		</tr>
		<tr>
		<td>withEntries</td>
		<td>[Int]</td>
		<td>an array of indices for data point input vectors to copy into this data set</td>
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
		<td>DataTypeError.invalidDataType</td>
		<td>thrown if the data set passed in is not a type that works with the data set the method is called on</td>
		</tr>
		<tr>
		<td>DataTypeError.wrongDimensionOnInput</td>
		<td>thrown if the input dimension for the data set passed in does not match the data set the method is called on</td>
		</tr>
		<tr>
		<td>DataIndexError.negative</td>
		<td>thrown if an index in the passed in array slice is negative</td>
		</tr>
		<tr>
		<td>DataIndexError.indexAboveDataSetSize</td>
		<td>thrown if an index in the passed in array slice is above the last data point index of the passed in data set</td>
		</tr>
		</table>
	</td>
</tr>
</table>

<table border="1">
<tr>
	<td>Template</td>
	<td>includeEntryInputs(fromDataSet: MLDataSet, withEntries: ArraySlice&lt;Int&gt;) throws</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method adds the entry inputs from the specified data set that match the array slice of indices provide to this data set.  Outputs ar set to zero</td>
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
		<td>fromDataSet</td>
		<td>MLDataSet</td>
		<td>the data set to get the new entries from</td>
		</tr>
		<tr>
		<td>withEntries</td>
		<td>ArraySlice&lt;Int&gt;</td>
		<td>an array slice of indices for data point input vectors to copy into this data set</td>
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
		<td>DataTypeError.invalidDataType</td>
		<td>thrown if the data set passed in is not a type that works with the data set the method is called on</td>
		</tr>
		<tr>
		<td>DataTypeError.wrongDimensionOnInput</td>
		<td>thrown if the input dimension for the data set passed in does not match the data set the method is called on</td>
		</tr>
		<tr>
		<td>DataIndexError.negative</td>
		<td>thrown if an index in the passed in array slice is negative</td>
		</tr>
		<tr>
		<td>DataIndexError.indexAboveDataSetSize</td>
		<td>thrown if an index in the passed in array slice is above the last data point index of the passed in data set</td>
		</tr>
		</table>
	</td>
</tr>
</table>

###singleOutput
<table border="1">
<tr>
	<td>Template</td>
	<td>singleOutput(_ index: Int) -> Double?</td>
</tr>
<tr>
	<td>Description</td>
	<td>This method returns the single output value for the specified point index.  If the index is out of range, a nil is returned.  If the data type is '.Classification', the value returned is the class label converted to a double.  If more than one output value on a regression data set, only the first value is returned.</td>
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
		<td>index</td>
		<td>Int</td>
		<td>the index to get the output value for</td>
		</tr>
		</table>
	</td>
</tr>
<tr>
	<td>Output</td>
	<td>Double? - the output value for the specified index</td>
</tr>
<tr>
	<td>Throws</td>
	<td>No</td>
</tr>
</table>

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

##PGEpisode
The PGEpisode class manages an array of PGStep structures.  The steps represent an episode for a Policy-Gradient Reinforcement Learning problem.  Each episode can be used to train a neural network to optimize the actions taken in future episodes.

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


