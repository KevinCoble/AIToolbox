# AIToolbox - the Manual
This manual is for the AIToolbox framework.  This framework is collection of Artificial Intelligence and Machine Learning algorithms, written in Swift (version 3.0)
### Using this manual
There is a lot to the AIToolbox framework, so this manual (when completed) will be quite large.  Lists of classes and methods are needed (the basic API reference), but also examples and how-to discussions.  This section of the manual gives some general overview with examples of how to write code for many of the more common operations.  The full API details are given in the Class, Enumerations, and Protocols sections.  Additional examples can be found in the XCTest files in the AIToolboxTests portion of the project.

##  Contents
[Using AIToolbox](#UsingAIToolbox)

* [Installation](#Installation)
* [Adding AIToolbox to Your Project](#Reference)
* [Machine Learning Types](#LearningTypes)

[Using a Regression Algorithm](#UseRegression)

* [Creating a Training Data Set](#TrainData)
* [Creating a Learning Class](#CreateClass)
* [Training the Learning Model](#TrainModel)
* [Creating a Testing or Use Data Set](#TestData)
* [Getting the Test/Use Results](#RegressionResults)

[Using a Classification Algorithm](#UseClassification)

* [Creating a Training Data Set](#TrainClassData)
* [Creating a Learning Class](#CreateClassClass)
* [Training the Learning Model](#TrainClassModel)
* [Creating a Testing or Use Data Set](#TestClassData)
* [Getting the Test/Use Results](#ClassificationResults)

[Using a Neural Network](#UseNN)

* [Creating a Neural Network](#CreateNN)
* [Initializing a Neural Network](#InitNN)
* [Training a Neural Network](#TrainNN)
* [Using a Trained Neural Network](#UsingNN)

[Classes](Classes.md)

[Structures](Structures.md)

[Enumerations](Enumerations.md)

[Protocols](Protocols.md)

## <a name="UsingAIToolbox"></a>Using AIToolbox
### <a name="Installation"></a>Installation
The framework comes as source code from GitHub (https://github.com/KevinCoble/AIToolbox) and must be built on your system.  It is possible to take just the source files you need for the algorithms you are going to use and add them to your project, but installing the complete framework into a system Framework directory will allow you to use all of the algorithms without including source code and determining which related files are needed for the classes you are going to use.
To build the code open the project, select the scheme to be built (AIToolbox or AIToolboxIOS) from the Project->Scheme menu, and type Command-B (If you are able to use this framework, you probably know how to build projects in XCode!).
If using this for macOS project you will want to install the framework into the system frameworks directory.  To get a copy of the framework (after building), go to the Project Navigator pane and open the 'Products' drop down list.  While holding the Option key down (to make a copy, not an alias), drag the AIToolbox.framework file (the top listing is for macOS) to its destination. 
Copy the framework into the /Library/Frameworks directory on your system.  You will need authentication to do this.
### <a name="Reference"></a>Adding AIToolbox to Your Project
####MacOS
To add AIToolbox to your project, go to the Project Settings and find the 'Linked Frameworks and Libraries' section.  Click on the '+' button to add the reference.  The framework selection dialog will appear.  Click on the 'Add Other' button to browse to the framework in the directory you installed it into (suggested to be /Library/Frameworks), and click the 'Open' button.  The framework can now be referenced by your application.
####iOS
This process puts the AIToolbox framework as an embedded binary in your application.

Have both the AIToolbox and your application projects open in XCode.  Make sure the AIToolbox scheme for iOS has been successfully built.

Right-click on the root application node in the project navigator. Click Add Files to “yourapplication”. In the file chooser, navigate to and select AIToolbox.xcodeproj. This will add AIToolbox.xcodeproj as a sub-project.

Expand the AIToolbox project to see the Products folder, and then look for for AIToolbox.framework beneath it. This file is the output of the framework project that packages up the binary code, headers, resources and metadata.  There are three outputs.  The iOS framework is the middle one.

In your application project, select the top level application node to open the project editor. Click the application target, and then go to the General tab.  Scroll down to the Embedded Binaries section.

Drag AIToolbox.framework (the middle one!) from the Products folder of AIToolbox.xcodeproj onto this section.  It will ask about grouping and references.  Select “copy items if needed” and click “Finish”.
####Referencing the framework
In each file that uses AIToolbox objects, you will need to import the framework.  Add the following line near the top of the file:
```
import AIToolbox
```
###  <a name="LearningTypes"></a>Machine Learning Types
There are three major types of machine learning algorithms (and a handful of other algorithms that may not fall into one of these categories).  They are regression algorithms, classification algorithms, and reinforcement learning.  

Algorithm | Information Type
------- | -------
Regression | Takes an input array and produces one or more real values.  Regression algorithms work like a function approximator
Classification | Takes an input array and returns a single integer that identifies the 'class' that the input vector best indicates
Reinforcement | Takes a series of states, actions, and rewards, and learns the best 'policy', or action to take from each state
There are several regression and classification algorithms in AIToolbox (there are several reinforcement learning algorithms too, but they use a different data type so are discussed separately).  To make using and comparing results easy, AIToolbox uses a protocol for specifying the input data, and provides a general-purpose **DataSet** class that conforms to the protocol.
There are also regression and classification protocols that the algorithms conform to, so once you know how to use one algorithm, the others of the same type will use the same calls.  These protocols are discussed in length in the protocol section.  The **DataSet** class is detailed in the classes documentation.  An example of creating a training data set, teaching an algorithm, and using it on test data is given below:
## <a name="UseRegression"></a>Using a Regression Algorithm
Using a regression algorithm consists of five parts, creating a training data set, creating and initializing an instance of the regression algorithm class, training the algorithm, creating a test/use data set, and using the regression class to get the results of that data set.  The following five sections show how these steps can be accomplished.  For further information, see the regression class you are using and the **Regressor** protocol API
### <a name="TrainData"></a>Creating a Training Data Set
Training of a supervised learning algorithm like regression is done using a data source object that conforms to the regression or classification protocol.  Since this example is using regression, we need a data source that implements the **MLRegressionDataSet** protocol.  The **DataSet** class provided by the framework fulfills this requirement.  Create a DataSet object of the correct type (.regression in this case), and give it the dimensions of the input and output vectors.  In this example we will be doing a simple linear regression of 1-dimension inputs to a 1-dimension output.  The following code snippet shows how to create the data set and add supervised learning data points to it:

```swift
        //  Create the training data
        let trainData = DataSet(dataType: .regression, inputDimension: 1, outputDimension: 1)
        do {
            try trainData.addDataPoint(input: [1.0], output: [8.3])
            try trainData.addDataPoint(input: [2.0], output: [11.0])
            try trainData.addDataPoint(input: [3.0], output: [14.7])
            try trainData.addDataPoint(input: [4.0], output: [19.7])
            try trainData.addDataPoint(input: [5.0], output: [26.7])
            try trainData.addDataPoint(input: [6.0], output: [35.2])
            try trainData.addDataPoint(input: [7.0], output: [44.4])
            try trainData.addDataPoint(input: [8.0], output: [55.9])
        }
        catch {
            print("error creating training data")
        }
```
### <a name="CreateClass"></a>Creating a Learning Class
Next you create an instance of the learning class.  We are using regression in this example, and there are several classes that provide regression.  Any class that conforms to the **Regressor** protocol can be used.  Some examples of classes that do this are **LinearRegressionModel**, **LogisticRegression**, **MixtureOfGaussians**, **NeuralNetwork**, and **NonLinearRegression**.  The following code snippet creates a linear regression class that takes a 1-dimensional input vector, and returns a 1-dimensional output vector (same as the data set created above), using a 1st order polygon (y = Ax+B)

```swift
        //  Create a model
        let lr = LinearRegressionModel(inputSize: 1, outputSize: 1, polygonOrder: 1)
```
        
### <a name="TrainModel"></a>Training the Learning Model
The next step is to use your training data to teach the learning class.  Some regression models can use multiple sets of data and can continue learning after the initial set is learned, while others need all data to be learned at once.  Linear regression needs all the data at once.  The following code snippet shows the previously created learning class being trained
```swift
        //  Train the model
        do {
            try lr.trainRegressor(trainData)
        }
        catch {
            print("Linear Regression Training error")
        }
```
### <a name="TestData"></a>Creating a Testing or Use Data Set
While supervised training data sets contain both input vectors and outputs, the data given to the model to test or use it have only inputs.  It is the outputs you want the model to calculate.  Therefore creating this data set is slightly different.  The following code snippet shows an example:

```swift
        //  Create a data set with the sequence
        let testData = DataSet(dataType: .regression, inputDimension: 1, outputDimension: 1)
        do {
            try testData.addTestDataPoint(input: [-5.0])
            try testData.addTestDataPoint(input: [9.0])
            try testData.addTestDataPoint(input: [17.0])
            try testData.addTestDataPoint(input: [0.0])
        }
        catch {
            print("Invalid test data set created")
        }
```
Note the input and output dimensions must match that expected by the learning class.
### <a name="RegressionResults"></a>Getting the Test/Use Results
The testing data is then given to the learning class to have it calculate the results.  The results are added to the test data object passed in.

```swift
        //  Get the results 
        do {
            try lr.predict(testData)
        }
        catch {
            print("Error having linear regression calculate results")
        }
        
        //  Use the results
        do {
            let result = try testData.getOutput(0)	//  Get first result
            ...
        }
        catch {
            print("Error getting results from data set")
        }
```
It is also possible to get a single result by passing the input vector in directly:

```swift
        //  Get the results 
        do {
            let result = try lr.predictOne([34.5])
            ...
        }
        catch {
            print("Error having linear regression calculate result")
        }
```

## <a name="UseClassification"></a>Using a Classification Algorithm
Using a classification algorithm consists of the same five parts that regression uses, but the data sets and training method names are different.  The following examples show the steps for classification learning.  For further information, see the regression class you are using and the **Classifier** protocol API.

### <a name="TrainClassData"></a>Creating a Training Data Set
Training of a supervised learning algorithm like regression is done using a data source object that conforms to the regression or classification protocol.  Since this example is using classification, we need a data source that implements the **MLClassificationDataSet** protocol.  The **DataSet** class provided by the framework fulfills this requirement.  Create a DataSet object of the correct type (.classification in this case), and give it the dimensions of the input vector.  The output vector size is ignored, but should be set to 1.  In this example we will be doing a simple support vector machine of 2-dimension inputs that lead to three possible classes.  The following code snippet shows how to create the data set and add supervised learning data points to it:

```swift
        //  Create the training data
        let trainData = DataSet(dataType: .classification, inputDimension: 2, outputDimension: 1)
        do {
            try trainData.addDataPoint(input: [0.2, 0.9], dataClass:0)
            try trainData.addDataPoint(input: [0.8, 0.3], dataClass:0)
            try trainData.addDataPoint(input: [0.5, 0.6], dataClass:0)
            try trainData.addDataPoint(input: [0.2, 0.7], dataClass:1)
            try trainData.addDataPoint(input: [0.2, 0.3], dataClass:1)
            try trainData.addDataPoint(input: [0.4, 0.5], dataClass:1)
            try trainData.addDataPoint(input: [0.5, 0.4], dataClass:2)
            try trainData.addDataPoint(input: [0.3, 0.2], dataClass:2)
            try trainData.addDataPoint(input: [0.7, 0.2], dataClass:2)
        }
        catch {
            print("error creating training data")
        }
```

### <a name="CreateClassClass"></a>Creating a Learning Class
Next you create an instance of the learning class.  We are using classification in this example, and there are several classes that provide classification.  Any class that conforms to the **Classifier** protocol can be used.  Some examples of classes that do this are **SVMModel**, **LogisticRegression**, and  **NeuralNetwork**.  The following code snippet creates a support vector machine class that takes a 2-dimensional input vector, and returns the class index (same as the data set created above)

```swift
        //  Create a model
        let svm = SVMModel(problemType: .c_SVM_Classification, kernelSettings:
            KernelParameters(type: .radialBasisFunction, degree: 0, gamma: 0.5, coef0: 0.0))
```
See the **SVMModel** class for more details on the input parameters such as configuring the kernel.
        
### <a name="TrainClassModel"></a>Training the Learning Model
The next step is to use your training data to teach the learning class.  Some classification models can use multiple sets of data and can continue learning after the initial set is learned, while others need all data to be learned at once.  Support vector machine models needs all the data at once.  The following code snippet shows the previously created learning class being trained.
```swift
        //  Train the model
        do {
            try svm.trainClassifier(trainData)
        }
        catch {
            print("SVM Training error")
        }
```
### <a name="TestClassData"></a>Creating a Testing or Use Data Set
While supervised training data sets contain both input vectors and output classes, the data given to the model to test or use it have only inputs.  It is the output class you want the model to determine.  Therefore creating this data set is slightly different.  The following code snippet shows an example:

```swift
        //  Create a data set with the sequence
        let testData = DataSet(dataType: .classification, inputDimension: 2, outputDimension: 1)
        do {
            try testData.addTestDataPoint(input: [0.7, 0.6])
            try testData.addTestDataPoint(input: [0.5, 0.7])
            try testData.addTestDataPoint(input: [0.1, 0.6])
            try testData.addTestDataPoint(input: [0.1, 0.4])
            try testData.addTestDataPoint(input: [0.3, 0.1])
            try testData.addTestDataPoint(input: [0.7, 0.1])
        }
        catch {
            print("Invalid test data set created")
        }
```
Note the input and output dimensions must match that expected by the learning class.

### <a name="ClassificationResults"></a>Getting the Test/Use Results
The testing data is then given to the learning class to have it calculate the results.  The results are added to the test data object passed in.

```swift
        //  Get the results 
        do {
            try svm.classify(testData)
        }
        catch {
            print("Error having SVM calculate results")
        }
        
        //  Use the results
        do {
            let result = try testData.getClass(0)	//  Get first result
            ...
        }
        catch {
            print("Error getting results from data set")
        }
```
It is also possible to get a single result by passing the input vector in directly:

```swift
        //  Get the results 
        do {
            let result = try svm.classifyOne([0.7, 0.6])
            ...
        }
        catch {
            print("Error having SVM calculate result")
        }
```

## <a name="UseNN"></a>Using a Neural Network
Using a Neural Network can have some differences from a standard regression or classification algorithm, although a neural network can be used in both.  A neural network ('NN' for short) can be trained on-line, meaning training can continue with new data after the initial training data has been performed.  This means that it is up to the code using the NN to tell it when to re-initialize itself.  The following sections outline creation, training, and use of a simple neural network
### <a name="CreateNN"></a>Creating a Neural Network
A Neural Network class contains a set of 'layers'.  The inputs to the network feed into the first layer.  The outputs of the first layer feed into the second layer, etc., until the final layer's outputs are the results from the NN.  To create a network, you create a set of tuples that define the characteristics of each layer, then pass this, along with the input vector dimension, to the init method for the network.  Each layer definition contains the type of layer, the number of nodes in the layer, the activation function for each node, and any auxiliary data needed by the layer.

```swift
        //  Create a 2 hidden layer - 2 node network, with one output node, using two inputs
        let layerDefs : [(layerType: NeuronLayerType, numNodes: Int, activation: NeuralActivationFunction, auxiliaryData: AnyObject?)] =
            [(layerType: .simpleFeedForward, numNodes: 2, activation: NeuralActivationFunction.hyperbolicTangent, auxiliaryData: nil),
             (layerType: .simpleFeedForward, numNodes: 2, activation: NeuralActivationFunction.hyperbolicTangent, auxiliaryData: nil),
             (layerType: .simpleFeedForward, numNodes: 1, activation: NeuralActivationFunction.hyperbolicTangent, auxiliaryData: nil)]
        let network = NeuralNetwork(numInputs: 2, layerDefinitions: layerDefs)
```
The layer type comes from the **NeuronLayerType** enumeration.  The activation function comes from the **NeuralActivationFunction** enumeration.
### <a name="InitNN"></a>Initializing a Neural Network
Since a neural network can be trained with more than a single set of data, it is up to the code using the network to tell it when to initialize the parameters within the layers or nodes.  The following code snippet shows the initialization:

```swift
        //  Initialize the weights
        network.initializeWeights(nil)
```
The initialize method takes as a parameter an optional data set that will be passed to the custom initialization function, if one has been set by calling the **setCustomInitializer** method first.  Passing nil results in the weights being initialized randomly.  Passing a data set results in the array return by the custom initializer method being used as the initial weights.
The initializaton routine is automatically called by the **trainClassifier** or **trainRegressor** methods, so is not needed to be explicitly called in those cases.
### <a name="TrainNN"></a>Training a Neural Network
Training a (non-recurrent) neural network is done in one of two ways, on-line or batch.  On-line training trains the network on a single input-vector-output vector/class index at a time, like in the following snippet:
```swift
        for index in 0..<numTrainingData {
            let input = (some way to get the training input vector)
            let expectedOutputs = (some way to get the training output vector)
            network.trainOne(input, expectedOutputs: [result], trainingRate: 0.5, weightDecay: 1.0)
        }
```
If using a the neural network as a classifier, the method would instead be **classificationTrainOne**.
To batch train a regression network, put your data into a regression or classification data set and use the following method:

```swift
            //  Train the batch
            network.batchTrain(data, epochIndices: trainIndexes, trainingRate: 0.5, weightDecay: 1.0)
```
The 'data' parameter is the data set, and the epochIndices parameter is an Int array of indices into the data to be used for the current batch.  A pair of methods are available that will use this method with regression or classification data, getting a random epochIndices array for each epoch of training:
```swift
SGDBatchTrain(_ trainData: MLRegressionDataSet, epochSize: Int, epochCount : Int, trainingRate: Double, weightDecay: Double)
classificationSGDBatchTrain(_ trainData: MLClassificationDataSet, epochSize: Int, epochCount : Int, trainingRate: Double, weightDecay: Double) throws
```
### <a name="UsingNN"></a>Using a Trained Neural Network
After training a neural network, it can be used the same as any other regression or classification algorithm, as outlined above.


