# Contributions
I made this framework public to share what I have learned, and to learn from others.  Contributions are gladly accepted.  This is a 'labor of love' of mine, so I reserve the right to be picky about what goes in.  With that in mind, here are a couple of suggestions if you want to contribute: 

## Ground Rules
I am trying to use the following guidelines for the code in the framework

  1.  The code should only use self-contained modules, or libraries supplied by Apple.  Even then, it would be best to not use proprietary Apple libraries much, so that the algorithms could be ported to other operating systems as Swift becomes more open.  LAPACK and vDSP (both part of Accelerate) are the two main libraries used.  Grand-Central Dispatch is also in the mix.
  2.  The code should try to match scholarly work on the algorithms where possible.  For example, almost all SVM discussions list the support vector lagrange multipliers as 'α', so I used that as the variable name - this is Swift, we can do that.
  3.  Most AI work seems to be done in Python.  I want to make Swift a first-class language for AI work.  Swift is generally faster than Python.  I try to take advantage of that fact and make these algorithms fast.
  4.  When possible, use existing modules rather than repeat work.  For example, Mixture-of-Gaussians can initialize using K-Means, so the KMeans class was used, rather than repeating the algorithm.  MDP used the Linear Regression module (now can use any Regressor)


## Requirements
I want to stick to the following rules for additions:

  1.  The code gets tinkered with a lot.  I want test classes for most algorithms to verify things haven't been broken when something gets changed.  These have to be short and simple - a 2 hour training run on an RNN just won't work for a full test.  They are also very useful as examples of how to use the algorithms, so they should be well written and well commented.
  2.  The code is Swift 3.0 compliant.  All new modules should also use this version of Swift
  3.  Base algorithm classes should be configurable, rather than subclassed or repeated:  i.e. no NeuralNetLayerTanh, NeuralNetLayerSigmoid, NeuralNetLayerRectLinear classes - the nonlinearity is configurable.
  4.  Good coding practices - well commented, clearly indented, etc. (Of course, I could use more attention to this myself!)
  5.  If the algorithm uses a set of data, it must take a data protocol implementation, not just an array of numbers
  6.  If the algorithm classifies input sets, it should adhere to the Classifier protocol
  7.  If the algorithm performs any sort of regression, it should adhere to the Regressor protocol.  If possible, it should regress multiple values in the output at once (i.e. a NN can have more than 1 node in the output layer)
  8.  All work should use Double precision values.  The one exception is the MetalNeuralNetwork class.  Unfortunately, Metal does not (yet) support double precision floats.


## To-Do List
Here is a list of things I would like to see added to the library.  Items marked with an 'ktc' are ones I have started playing with, or hope to get to soon.  This does not mean you cannot contribute that item yourself - I unfortunately have limited time to work on this, and my attention often wanders (squirrel!)

  1.  **Algorithms**
    * GRU Recurrent Network layers
    * More non-linear regression solutions (I had started on a Levenberg-Marquardt solver, but went on to Validation instead)
    * Locally weighted least-squares
    * Weight dropping in Neural Networks
    * Bayesian Regression
    * Boosting
    * MDP variants - partially observable, other solutions for the continuous-state versions, LQR 
    * Heuristic search algorithms
    * Natural Language Processing? (I have done work in this, but not in any publishable context)

  2.  **Optimization**
    *  Benchmarking
    *  Profiling and optimizing the code
    *  Explore more parallelization using GCD
    *  NN Layers without node classes (matrix weights, etc.)
    *  More use of Metal (although this obscures the algorithms)

  3.  **Data I/O**
    *  Persistance for all models (write and read parameters to/from persistant storage).  SVM has a version of this now
    *  Subclass of DataSet that uses sparse storage (have added protocols to allow this)
    *  Possible add-ins to read data from publicly known data set types

  4.  **Data Viewing**
    *  Interactions (mouse/keyboard/touch) for view (scaling, hiding legend, changing data source indexes, changing unplotted input values)

  5.  **Ease-of-Use additions**
    *  Manual - started, could always be improved
    *  Better markup in the classes for use patterns and auto-complete
    *  Playgrounds showing the use of some of the algorithms for education purposes (three created!  Many more needed)

And this is just what came to mind for now.  If you have other things you would like to see added or changed, you can email me at kevinc@macrobotics.com or open an issue

If you use the framework in a program you have made public, contact me and I'll be glad to put a link to it in this repository.


