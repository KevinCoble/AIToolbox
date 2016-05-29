//
//  SVM.swift
//  AIToolbox
//
//  Created by Kevin Coble on 10/6/15.
//
//  Based on LIBSVM
//  Copyright (c) 2000-2014 Chih-Chung Chang and Chih-Jen Lin
//  All rights reserved
//  See LIBSVM copyright notice for details
//

import Foundation

public enum SVMType: Int     //  SVM problem type
{
    case C_SVM_Classification = 0
    case ν_SVM_Classification
    case OneClassSVM
    case ϵSVMRegression
    case νSVMRegression
}

struct DecisionFunction {
    var ρ : Double
    var α : [Double]
}


public class SVMModel
{
    ///  Parameters to be set by the caller
    let type : SVMType          //  Type of SVM problem we are trying to solve
    public var Cost : Double = 1.0     //  Cost parameter for for C_SVM_Classification, ϵSVMRegression and νSVMRegression
    public var weightModifiers : [(classLabel: Int, multiplier: Double)]?    //  Cost modifiers for each class
    public var probability = false     //  Flag indicating probabilities should be calculated
    public var kernelParams: KernelParameters
    public var ϵ = 1e-3                // stopping criteria
    public var ν = 0.5                 //  For ν classification and regression
    public var p = 0.1                 //  for ϵ regression
    
    //  Internal storage
    
    //  Solution results
    var numClasses = 0
    var labels: [Int] = []
    var ρ : [Double] = []
    var totalSupportVectors = 0
    var supportVectorCount: [Int] = []
    var supportVector: [[Double]] = []
    var coefficients: [[Double]] = []       //  Array for each class, element for each support vector, ordered by opposing class index
    var probabilityA: [Double] = []
    var probabilityB: [Double] = []
    
    ///  Create an SVM model
    public init(problemType : SVMType, kernelSettings: KernelParameters)
    {
        type = problemType
        
        //  Initialize the kernal parameters to a linear kernal
        kernelParams = kernelSettings
    }
    
    public init(copyFrom: SVMModel)
    {
        type = copyFrom.type
        kernelParams = copyFrom.kernelParams
        Cost = copyFrom.Cost
        weightModifiers = copyFrom.weightModifiers
        probability = copyFrom.probability
    }
    
    public init?(loadFromFile path: String)
    {
        //  Initialize all the stored properties (Swift requires this, even when returning nil [supposedly fixed in Swift 2.2)
        numClasses = 0
        labels = []
        ρ = []
        totalSupportVectors = 0
        supportVectorCount = []
        supportVector = []
        coefficients = []
        probabilityA = []
        probabilityB = []
        kernelParams = KernelParameters(type: .RadialBasisFunction, degree: 0, gamma: 0.5, coef0: 0.0)
        
        //  Read the property list
        let pList = NSDictionary(contentsOfFile: path)
        if pList == nil {type = .C_SVM_Classification; return nil }
        let dictionary : Dictionary = pList! as! Dictionary<String, AnyObject>
        
        //  Get the training results from the dictionary
        let typeValue = dictionary["type"] as? NSInteger
        if typeValue == nil {type = .C_SVM_Classification; return nil }
        let testType = SVMType(rawValue: typeValue!)
        if testType == nil {type = .C_SVM_Classification; return nil }
        type = testType!
        
        let numClassValue = dictionary["numClasses"] as? NSInteger
        if numClassValue == nil { return nil }
        numClasses = numClassValue!
        
        let labelArray = dictionary["labels"] as? NSArray
        if labelArray == nil { return nil }
        labels = labelArray! as! [Int]
        
        let rhoArray = dictionary["ρ"] as? NSArray
        if rhoArray == nil { return nil }
        ρ = rhoArray! as! [Double]
        
        let totalSVValue = dictionary["totalSupportVectors"] as? NSInteger
        if totalSVValue == nil { return nil }
        totalSupportVectors = totalSVValue!
        
        let svCountArray = dictionary["supportVectorCount"] as? NSArray
        if svCountArray == nil { return nil }
        supportVectorCount = svCountArray! as! [Int]
        
        let svArray = dictionary["supportVector"] as? NSArray
        if svArray == nil { return nil }
        supportVector = svArray! as! [[Double]]
        
        let coeffArray = dictionary["coefficients"] as? NSArray
        if coeffArray == nil { return nil }
        coefficients = coeffArray! as! [[Double]]
        
        let probAArray = dictionary["probabilityA"] as? NSArray
        if probAArray == nil { return nil }
        probabilityA = probAArray! as! [Double]
        
        let probBArray = dictionary["probabilityB"] as? NSArray
        if probBArray == nil { return nil }
        probabilityB = probBArray! as! [Double]
    }
    
    public func isνFeasableForData(data: DataSet) -> Bool
    {
        if (type != .ν_SVM_Classification) { return true }
        
        var label : [Int] = []
        var count : [Int] = []
        for i in 0..<data.size {
            if let index = label.indexOf(data.classes![i]) {
                //  label already found
                count[index] += 1
            }
            else {
                //  new label - add to list
                label.append(data.classes![i])
                count.append(1)
            }
        }
        
        for i in 0..<label.count {
            for j in (i+1)..<label.count {
                if (ν * Double(count[i] + count[j]) * 0.5 > Double(min(count[i], count[j]))) {
                    return false
                }
            }
        }
        
        return true
    }
    
    ///  Method to 'train' the SVM
    public func train(data: DataSet)
    {
        //  Training depends on the problem type
        switch (type) {
            //  Training for one-class classification or regression
        case .OneClassSVM, .ϵSVMRegression, .νSVMRegression:
            
            //  Initialize the model parameters for single-class
            numClasses = 1
            labels = []
            supportVector = []
            coefficients = [[]]
            
            //  If probability flag on, calculate the probabilities
            if (probability && (type == .ϵSVMRegression || type == .νSVMRegression)) {
                probabilityA = [svrProbability(data)]
            }
            
            //  Train one set of support vectors
            let f = trainOne(data, costPositive: 0.0, costNegative: 0.0, display: true)
            ρ = [f.ρ]
            
            //  Build the output
            totalSupportVectors = 0
            supportVector = []
            coefficients = [[]]
            for index in 0..<data.size {    //  Get the support vector points and save them
                if (fabs(f.α[index]) > 0.0) {
                    totalSupportVectors += 1
                    supportVector.append(data.inputs[index])
                    coefficients[0].append(f.α[index])
                }
            }
            
            break
        
            //  Training for classification
        case .C_SVM_Classification, .ν_SVM_Classification:
            //  Group training data of the same class
            do {
                //  Group the data into classes
                try data.groupClasses()
                let classificationData = data.optionalData as! ClassificationData
                if (classificationData.numClasses <= 1) {
                    print("Invalid number of classes in data")
                    return
                }
                
                //  Calculate weighted Cost for each class label
                var weightedCost = [Double](count: classificationData.numClasses, repeatedValue:Cost)
                if let weightMods = weightModifiers {
                    for mod in weightMods {
                        let index = classificationData.foundLabels.indexOf(mod.classLabel)
                        if (index == nil) {
                            print("weight modifier label \(mod.classLabel) not found in data set")
                            continue
                        }
                        else {
                            weightedCost[index!] *= mod.multiplier
                        }
                    }
                }
                
                //  Train numClasses * (numClasses - 1) / 2 models
                var nonZero = [Bool](count: data.size, repeatedValue: false)
                var functions: [DecisionFunction] = []
                probabilityA = []
                probabilityB = []
                for i in 0..<classificationData.numClasses-1 {
                    for j in i+1..<classificationData.numClasses {
                        //  Create a sub-problem data set with just the i and j class data
                        if let subProblem = DataSet(fromDataSet: data, withEntries: classificationData.classOffsets[i]) {
                            do {
                                try subProblem.includeEntries(fromDataSet: data, withEntries: classificationData.classOffsets[j])
                                
                                //  Set the sub-problem class labels to 1 and -1
                                for index in 0..<classificationData.classCount[i] {
                                    try subProblem.setClass(index, newClass: 1)
                                }
                                for index in classificationData.classCount[i]..<subProblem.size {
                                    try subProblem.setClass(index, newClass: -1)
                                }
                                
                                //  If the probability flag is set, calculate the probabilities
                                if (probability) {
                                    let result = binarySVCProbability(subProblem, positiveLabel: classificationData.foundLabels[i], costPositive: weightedCost[i], costNegative: weightedCost[j])
                                    probabilityA.append(result.A)
                                    probabilityB.append(result.B)
                                }
                                
                                //  Train on this sub-problem
                                let f = trainOne(subProblem, costPositive: weightedCost[i], costNegative: weightedCost[j], display: true)
                                functions.append(f)
                                
                                //  Mark the non-zero α's
                                for index in 0..<classificationData.classCount[i] {
                                    if (fabs(f.α[index]) > 0.0) { nonZero[classificationData.classOffsets[i][index]] = true}
                                }
                                for index in 0..<classificationData.classCount[j] {
                                    if (fabs(f.α[index+classificationData.classCount[i]]) > 0.0) { nonZero[classificationData.classOffsets[j][index]] = true}
                                }
                            }
                        }
                    }
                }
                
                //  Build output
                numClasses = classificationData.numClasses
                labels = classificationData.foundLabels

                ρ = []
                for df in functions {
                    ρ.append(df.ρ)
                }

                totalSupportVectors = 0
                supportVectorCount = [Int](count: classificationData.numClasses, repeatedValue: 0)
                for i in 0..<classificationData.numClasses {
                    for j in 0..<classificationData.classCount[i] {
                        if(nonZero[classificationData.classOffsets[i][j]]) {
                            supportVectorCount[i] += 1
                            totalSupportVectors += 1
                        }
                    }
                }
                print("Total nSV = \(totalSupportVectors)")
                supportVector = []
                for index in 0..<data.size {    //  Get the support vector points and save them
                    if (nonZero[index]) {
                        supportVector.append(data.inputs[index])
                    }
                }
                
                //  Get the start locations in the coefficient array for each class's coeffiecient
                var coeffStart: [Int] = [0]
                for index in 0..<classificationData.numClasses-1 {
                    coeffStart.append(coeffStart[index] + supportVectorCount[index])
                }
                
                //  Save the α's for each class permutation as a set of coefficients of the support vectors
                coefficients = []
                for _ in 0..<classificationData.numClasses-1 {
                    coefficients.append([Double](count:totalSupportVectors, repeatedValue: 0.0))
                }
                var permutation = 0
                for i in 0..<classificationData.numClasses-1 {
                    for j in i+1..<classificationData.numClasses {
                        var q = coeffStart[i]
                        for index in 0..<classificationData.classCount[i] {
                            if (nonZero[classificationData.classOffsets[i][index]]) {
                                coefficients[j-1][q] = functions[permutation].α[index]
                                q += 1
                            }
                        }
                        q = coeffStart[j]
                        for index in 0..<classificationData.classCount[j] {
                            if (nonZero[classificationData.classOffsets[j][index]]) {
                                coefficients[i][q] = functions[permutation].α[index + classificationData.classCount[i]]
                                q += 1
                            }
                        }
                        permutation += 1
                    }
                }
            }
            catch {
                //  Handle error
            }
            break
        }
    }
    
    private func trainOne(data: DataSet, costPositive : Double, costNegative : Double, display : Bool = false) -> DecisionFunction
    {
        var solver : Solver?
        
        //  Use the solver to determine the support vectors
        switch (type) {
        case .C_SVM_Classification:
            //  Instantiate a solver class
            solver = Solver(kernelParams: kernelParams, ϵ: ϵ)
            solver!.costPositive = costPositive
            solver!.costNegative = costNegative
            
            solver!.solveClassification(data, display: display)
            break
        case .ν_SVM_Classification:
            //  Instantiate a solver class
            solver = Solver_ν(kernelParams: kernelParams, ϵ: ϵ, ν: ν)
            solver!.costPositive = 1.0
            solver!.costNegative = 1.0
           
            solver!.solveClassification(data, display: display)
            break
        case .OneClassSVM:
            //  Instantiate a solver class
            solver = Solver(kernelParams: kernelParams, ϵ: ϵ)
            solver!.costPositive = 1.0
            solver!.costNegative = 1.0
            
            solver!.solveOneClass(data, ν: ν, display: display)
            break
        case .ϵSVMRegression:
            //  Instantiate a solver class
            solver = Solver(kernelParams: kernelParams, ϵ: ϵ)
            solver!.costPositive = Cost
            solver!.costNegative = Cost
            
            solver!.solveRegression(data, p: p, display: display)
            break
        case .νSVMRegression:
            //  Instantiate a solver class
            solver = Solver_ν(kernelParams: kernelParams, ϵ: ϵ, ν: ν)
            solver!.costPositive = Cost
            solver!.costNegative = Cost
            
            solver!.solveRegression(data, p: p, display: display)
            break
        }
        
        //  If display flag is set, show the results
        if (display) {
            print("obj = \(solver!.obj), rho = \(solver!.ρ)")
            
            //  Count the number of support vectors
            var numSupportVectors = 0
            var numBaseSupportVectors = 0
            for index in 0..<data.size {
                let entry = solver!.α[index]
                if(abs(entry) > 0.0)
                {
                    numSupportVectors += 1
                    if let output = data.singleOutput(index) {
                        if(output > 0.0) {
                            if(abs(entry) >= solver!.positiveUpperBound) {numBaseSupportVectors += 1}
                        }
                        else {
                            if(abs(entry) >= solver!.negativeUpperBound) {numBaseSupportVectors += 1}
                        }
                    }
                }
            }
            print("nSV = \(numSupportVectors), nBSV = \(numBaseSupportVectors)");
        }
        
        let f = DecisionFunction(ρ: solver!.ρ, α: solver!.α)
        return f
    }
    
    public func crossValidation(data: DataSet, numberOfFolds: Int) -> [Double]
    {
        var target = [Double](count: data.size, repeatedValue: 0.0)
        
        //  Limit check the number of folds
        var nFolds = numberOfFolds
        if (nFolds > data.size)
        {
            nFolds = data.size
            print("WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)")
        }

        //  Get the fold data set indices
        var foldStart : [Int] = []
        var perm : [Int] = []
        if ((type == .C_SVM_Classification || type == .ν_SVM_Classification) && nFolds < data.size) {
            //  Group the classes
            do {
                //  Group the data into classes
                try data.groupClasses()
                let classificationData = data.optionalData as! ClassificationData
                
                //  Get a random shuffle of the data in each class
                var shuffledIndices = classificationData.classOffsets
                for c in 0..<classificationData.numClasses {
                    for i in 0..<classificationData.classCount[c] {
                        let j =  i + Int(rand()) % (classificationData.classCount[c] - i)
                        swap(&shuffledIndices[c][i], &shuffledIndices[c][j])
                    }
                }
                
                //  Get the count of items for each fold
                var foldCount = [Int](count: nFolds, repeatedValue: 0)
                for i in 0..<nFolds {
                    for c in 0..<classificationData.numClasses {
                        foldCount[i] += (i+1) * classificationData.classCount[c] / nFolds - i * classificationData.classCount[c] / nFolds
                    }
                }
                
                //  Get the start of each fold
                foldStart.append(0)
                for i in 1...nFolds {
                    foldStart.append(foldStart[i-1] + foldCount[i-1])
                }
                
                //  Get the permutation index array
                perm = []
                for c in 0..<classificationData.numClasses {
                    for i in 0..<nFolds {
                        let begin = i * classificationData.classCount[c] / nFolds;
                        let end = (i+1) * classificationData.classCount[c] / nFolds;
                        for j in begin..<end {
                            perm.append(shuffledIndices[c][j])
                        }
                    }
                }
            }
            catch {
                //  Handle error
            }
            
        }
        else {
            perm = data.getRandomIndexSet()
            for i in 0...nFolds {
                foldStart.append(i * data.size / nFolds)
            }
        }
        
        //  Calculate each fold
        for i in 0..<nFolds {
            //  Create the sub-problem data set, all points except those in the fold
            var subIndices : [Int] = []
            if (i > 0) {
                subIndices += Array(perm[0..<foldStart[i]])
            }
            if (i < nFolds) {
                subIndices += Array(perm[foldStart[i]]..<data.size)
            }
            if let subProblem = DataSet(fromDataSet: data, withEntries: subIndices) {
                let subModel = SVMModel(copyFrom:self)
                subModel.train(subProblem)
                if (probability && (type == .C_SVM_Classification || type == .ν_SVM_Classification)) {
                    for j in foldStart[i]..<foldStart[i+1] {
                        target[perm[j]] = subModel.predictProbability(data.inputs[perm[j]])
                    }
                }
                else {
                    for j in foldStart[i]..<foldStart[i+1] {
                        target[perm[j]] = subModel.predictOne(data.inputs[perm[j]])
                    }
                }
            }
        }
        
        return target
    }
    
    func binarySVCProbability(data: DataSet, positiveLabel: Int, costPositive: Double, costNegative: Double) -> (A: Double, B: Double)
    {
        //  Get a shuffled index set
        var perm = data.getRandomIndexSet()
        
        //  Create the array for the decision values
        var decisionValues = [Double](count: data.size, repeatedValue: 0.0)
        
        //  Do 5 cross-validations
        let nr_fold = 5
        for index in 0..<nr_fold {
            //  Create a sub-problem with the data without the cross validation set
            let begin = index * data.size / nr_fold;
            let end = (index+1) * data.size / nr_fold;
            var subIndices = Array(perm[0..<begin])
            subIndices += Array(perm[end..<data.size])
            if let subProblem = DataSet(fromDataSet: data, withEntries: subIndices) {
                //  Count the number of positive and negative data points in the sub-problem
                var countPositive = 0
                var countNegative = 0
                var positiveLabel = 0
                var negativeLabel = 0
                for index in 0..<subProblem.size {
                    if (subProblem.classes![index] == positiveLabel) {
                        countPositive += 1
                        positiveLabel = subProblem.classes![index]
                    }
                    else {
                        countNegative += 1
                        negativeLabel = subProblem.classes![index]
                    }
                }
                
                //  Set the decision values
                if (countPositive==0 && countNegative==0) {
                    for index in begin..<end { decisionValues[perm[index]] = 0.0 }
                }
                else if(countPositive > 0 && countNegative == 0) {
                    for index in begin..<end { decisionValues[perm[index]] = 1.0 }
                }
                else if(countPositive == 0 && countNegative > 0) {
                    for index in begin..<end { decisionValues[perm[index]] = -1.0 }
                }
                else
                {
                    //  Train an SVM on the sub-problem
                    let subModel = SVMModel(problemType: type, kernelSettings: kernelParams)
                    subModel.weightModifiers = [(classLabel: positiveLabel, multiplier: costPositive), (classLabel: negativeLabel, multiplier: costNegative)]
                    subModel.train(subProblem)
                    
                    //  Set the decision values based on the predictions from the sub-model
                    for index in begin..<end {
                        decisionValues[perm[index]] = Double(subModel.predictOneFromBinaryClass(data.inputs[index]))
                    }
                }
            }
        }
        
        return sigmoidTrain(decisionValues, labels: data.classes!)
    }

    func sigmoidTrain(decisionValues: [Double], labels: [Int]) -> (A: Double, B: Double)
    {
        //  Count the prior labels
        var prior0 = 0.0
        var prior1 = 0.0
        for label in labels {
            if (label > 0) {
                prior1 += 1.0
            }
            else {
                prior0 += 1.0
            }
        }
        
        //  Set up iteration parameters
        let max_iter = 100	// Maximal number of iterations
        let min_step = 1e-10	// Minimal step taken in line search
        let sigma = 1e-12     // For numerically strict PD of Hessian
        let eps = 1e-5
        let hiTarget = (prior1+1.0)/(prior1+2.0)
        let loTarget = 1/(prior0+2.0)
        
        //  Initialize Point and Initial Fun Value
        var A = 0.0
        var B = log((prior0+1.0)/(prior1+1.0))
        var fval = 0.0
        var t : [Double] = []

        for index in 0..<labels.count {
            if (labels[index] > 0) {
                t.append(hiTarget)
            }
            else {
                t.append(loTarget)
            }
            let fApB = decisionValues[index] * A + B
            if (fApB>=0) {
                fval += t[index] * fApB + log(1+exp(-fApB))
            }
            else {
                fval += (t[index] - 1.0) * fApB + log(1.0 + exp(fApB))
            }
        }
        
        for iter in 0..<max_iter {
            // Update Gradient and Hessian (use H' = H + sigma I)
            var h11 = sigma // numerically ensures strict PD
            var h22 = sigma
            var h21 = 0.0
            var g1 = 0.0
            var g2 = 0.0
            for index in 0..<labels.count {
                let fApB = decisionValues[index] * A + B
                var p, q : Double
                if (fApB >= 0) {
                    p=exp(-fApB)/(1.0+exp(-fApB));
                    q=1.0/(1.0+exp(-fApB));
                }
                else {
                    p=1.0/(1.0+exp(fApB));
                    q=exp(fApB)/(1.0+exp(fApB));
                }
                let d2 = p * q
                h11 += decisionValues[index] * decisionValues[index] * d2
                h22 += d2
                h21 += decisionValues[index] * d2
                let d1 = t[index] - p
                g1 += decisionValues[index] * d1
                g2 += d1
            }
            
            // Stopping Criteria
            if (fabs(g1)<eps && fabs(g2)<eps) { break }
            
            // Finding Newton direction: -inv(H') * g
            let det = h11 * h22 - h21 * h21;
            let dA = -(h22 * g1 - h21 * g2) / det
            let dB = -(-h21 * g1 + h11 * g2) / det
            let gd = g1 * dA + g2 * dB
            
            var stepsize = 1.0		// Line Search
            while (stepsize >= min_step)
            {
                let newA = A + stepsize * dA
                let newB = B + stepsize * dB
                
                // New function value
                var newf = 0.0
                for index in 0..<labels.count {
                    let fApB = decisionValues[index] * newA + newB
                    if (fApB >= 0) {
                        newf += t[index]*fApB + log(1+exp(-fApB))
                    }
                    else {
                        newf += (t[index] - 1.0) * fApB + log(1.0 + exp(fApB))
                    }
                }
                // Check sufficient decrease
                if (newf < fval+0.0001*stepsize*gd)
                {
                    A = newA
                    B = newB
                    fval = newf
                    break
                }
                else {
                    stepsize = stepsize * 0.5
                }
            }
            if (stepsize < min_step) { print("Line search fails in two-class probability estimates") }
            
            if iter >= max_iter { print("Reaching maximal iterations in two-class probability estimates") }
        }
        
        return (A: A, B: B)
    }
    
    func svrProbability(data: DataSet) -> Double
    {
        //  Run cross-validation without calculating probabilities
        let oldProbabilityFlag = probability
        probability = false
        var ymv = crossValidation(data, numberOfFolds: 5)
        probability = oldProbabilityFlag
        
        //  Calculate the final probability estimate
        var mae = 0.0
        for i in 0..<data.size {
            ymv[i] = data.outputs![i][0] - ymv[i]
            mae += fabs(ymv[i])
        }		
        mae /= Double(data.size)
        let std = sqrt(2 * mae * mae)
        var count=0
        mae=0.0
        for i in 0..<data.size {
            if (fabs(ymv[i]) > 5*std) {
                count += 1
            }
            else {
                mae += fabs(ymv[i])
            }
        }
        mae /= Double(data.size-count)
        print("Prob. model for test data: target value = predicted value + z, z: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma= \(mae)")
        return mae
    }
    
    public func predictValues(data: DataSet)
    {
        //  Initialize the output variables
        data.outputs = []
        if (type == .C_SVM_Classification || type == .ν_SVM_Classification) {data.classes = []}
        
        //  Get the support vector start index for each class
        var coeffStart = [0]
        for index in 0..<numClasses-1 {
            coeffStart.append(coeffStart[index] + supportVectorCount[index])
        }
        
        //  Determine each value based on the input
        for index in 0..<data.size {
            switch (type) {
                //  Predict for one-class classification or regression
            case .OneClassSVM, .ϵSVMRegression, .νSVMRegression:
                var sum = 0.0
                for i in 0..<totalSupportVectors {
                    let kernelValue = Kernel.calcKernelValue(kernelParams, x: data.inputs[index], y: supportVector[i])
                    sum += coefficients[0][i] * kernelValue
                }
                sum -= ρ[0]
                data.outputs!.append([sum])
                if (type == .OneClassSVM) {
                    data.classes!.append((sum>0) ? 1: -1)
                }
                break
                
                //  Predict for classification
            case .C_SVM_Classification, .ν_SVM_Classification:
                //  Get the kernel value for each support vector at the input value
                var kernelValue: [Double] = []
                for sv in 0..<totalSupportVectors {
                    kernelValue.append(Kernel.calcKernelValue(kernelParams, x: data.inputs[index], y: supportVector[sv]))
                }
                
                //  Allocate vote space for the classification
                var vote = [Int](count: numClasses, repeatedValue: 0)
                
                //  Initialize the decision value storage in the data set
                var decisionValues: [Double] = []
                
                //  Get the seperation info between each class pair
                var permutation = 0
                for i in 0..<numClasses {
                    for j in i+1..<numClasses {
                        var sum = 0.0
                        for k in 0..<supportVectorCount[i] {
                            sum += coefficients[j-1][coeffStart[i]+k] * kernelValue[coeffStart[i]+k]
                        }
                        for k in 0..<supportVectorCount[j] {
                            sum += coefficients[i][coeffStart[j]+k] * kernelValue[coeffStart[j]+k]
                        }
                        sum -= ρ[permutation]
                        decisionValues.append(sum)
                        permutation += 1
                        if (sum > 0) {
                            vote[i] += 1
                        }
                        else {
                            vote[j] += 1
                        }
                    }
                }
                data.outputs!.append(decisionValues)
                
                //  Get the most likely class, and set it
                var maxIndex = 0
                for index in 1..<numClasses {
                    if (vote[index] > vote[maxIndex]) { maxIndex = index }
                }
                data.classes!.append(labels[maxIndex])
                break
            }
        }
    }
    
    public func predictOne(inputs: [Double]) -> Double
    {
        //  Get the support vector start index for each class
        var coeffStart = [0]
        for index in 0..<numClasses-1 {
            coeffStart.append(coeffStart[index] + supportVectorCount[index])
        }
        //  Get the kernel value for each support vector at the input value
        var kernelValue: [Double] = []
        for sv in 0..<totalSupportVectors {
            kernelValue.append(Kernel.calcKernelValue(kernelParams, x: inputs, y: supportVector[sv]))
        }
        
        var sum = 0.0
        for k in 0..<supportVectorCount[0] {
            sum += coefficients[0][coeffStart[0]+k] * kernelValue[coeffStart[0]+k]
        }
        for k in 0..<supportVectorCount[1] {
            sum += coefficients[0][coeffStart[1]+k] * kernelValue[coeffStart[1]+k]
        }
        sum -= ρ[0]
        
        return sum
    }
    
    public func predictOneFromBinaryClass(inputs: [Double]) -> Int
    {
        let sum = predictOne(inputs)
        if (sum > 0) {
            return labels[0]
        }
        else {
            return labels[1]
        }
    }
    
    public func predictProbability(inputs: [Double]) -> Double
    {
        if ((type == .C_SVM_Classification || type == .ν_SVM_Classification) && probabilityA.count > 0 && probabilityB.count > 0) {
            let data = DataSet(dataType: .Classification, inputDimension: inputs.count, outputDimension: 1)
            do {
                try data.addDataPoint(input: inputs, output: [0.0])
            }
            catch {
                print("dimension error on inputs")
            }
            predictValues(data)
            let minProbability = 1e-7
            var pairwiseProbability : [[Double]] = []
            var k = 0
            for i in 0..<numClasses-1 {
                for j in i+1..<numClasses {
                    pairwiseProbability[i][j] = min(max(SVMModel.sigmoidPredict(data.outputs![0][k], probA: probabilityA[k], probB: probabilityB[k]), minProbability), 1.0-minProbability)
                    pairwiseProbability[j][i] = 1.0 - pairwiseProbability[i][j]
                    k += 1
                }
            }
            let probabilityEstimates = multiclassProbability(pairwiseProbability)
            var maxIndex = 0
            for i in 1..<numClasses {
                if (probabilityEstimates[i] > probabilityEstimates[maxIndex]) {
                    maxIndex = i
                }
            }
            return Double(labels[maxIndex])
        }
        else {
            return predictOne(inputs)
        }
    }
    
    class func sigmoidPredict(decisionValue: Double, probA: Double, probB: Double) -> Double
    {
        let fApB = decisionValue * probA + probB
        // 1-p used later; avoid catastrophic cancellation
        if (fApB >= 0) {
            return exp(-fApB) / (1.0 + exp(-fApB))
        }
        else {
            return 1.0 / (1.0 + exp(fApB))
        }
    }
    
    // Method 2 from the multiclass_prob paper by Wu, Lin, and Weng
    func multiclassProbability(probabilityPairs: [[Double]]) -> [Double]
    {
        var p : [Double] = []
        var Q : [[Double]] = []
        
        let k = probabilityPairs.count
        let eps = 0.005/Double(k)
        
        for t in 0..<k {
            p.append(1.0 / Double(k))
            Q.append([Double](count: k, repeatedValue: 0.0))
            for j in 0..<t {
                Q[t][t] += probabilityPairs[j][t] * probabilityPairs[j][t]
                Q[t][j] = Q[j][t]
            }
            for j in t+1..<k {
                Q[t][t] += probabilityPairs[j][t] * probabilityPairs[j][t]
                Q[t][j] = -probabilityPairs[j][t] * probabilityPairs[t][j]
            }
        }
        
        var iter = 0
        let max_iter = max(100,k)
        var pQp : Double
        var Qp = [Double](count: k, repeatedValue: 0.0)
        while (iter < max_iter) {
            // stopping condition, recalculate QP,pQP for numerical accuracy
            pQp=0.0
            for t in 0..<k {
                Qp[t]=0;
                for j in 0..<k {
                    Qp[t] += Q[t][j] * p[j]
                }
                pQp+=p[t] * Qp[t];
            }
            var max_error = 0.0
            for t in 0..<k {
                let error = fabs(Qp[t] - pQp)
                if (error > max_error) { max_error = error }
            }
            if (max_error < eps) { break }
            
            for t in 0..<k {
                let diff = (-Qp[t] + pQp) / Q[t][t];
                p[t] += diff
                pQp = (pQp + diff * (diff * Q[t][t] + 2.0 * Qp[t])) / (1.0 + diff) / (1.0 + diff)
                for j in 0..<k {
                    Qp[j] = (Qp[j] + diff * Q[t][j]) / (1.0 + diff)
                    p[j] /= (1.0 + diff)
                }
            }
            
            iter += 1
        }
        
        if (iter >= max_iter) {
            print("Exceeds max_iter in multiclass_prob")
        }
        
        return p
    }
    
    ///  Routine to write the model result parameters to a property list path at the provided path
    public enum SVMWriteErrors: ErrorType { case failedWriting }
    public func saveToFile(path: String) throws
    {
        //  Create a property list of the SVM model
        var modelDictionary = [String: AnyObject]()
        modelDictionary["type"] = type.rawValue
        modelDictionary["numClasses"] = numClasses
        modelDictionary["labels"] = labels
        modelDictionary["ρ"] = ρ
        modelDictionary["totalSupportVectors"] = totalSupportVectors
        modelDictionary["supportVectorCount"] = supportVectorCount
        modelDictionary["supportVector"] = supportVector
        modelDictionary["coefficients"] = coefficients
        modelDictionary["probabilityA"] = probabilityA
        modelDictionary["probabilityB"] = probabilityB
        
        //  Convert to a property list (NSDictionary) and write
        let pList = NSDictionary(dictionary: modelDictionary)
        if !pList.writeToFile(path, atomically: false) { throw SVMWriteErrors.failedWriting }
    }
}

//  Alpha state for QP solver
internal enum AlphaState {
    case LowerBound
    case UpperBound
    case Free
}

//  Internal class for solving the quadratic programming 

// An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
// Solves:
//
//	min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//		y^T \alpha = \delta
//		y_i = +1 or -1
//		0 <= alpha_i <= Cp for y_i = 1
//		0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//	Q, p, y, Cp, Cn, and an initial feasible point \alpha
//	l is the size of vectors and matrices
//	eps is the stopping tolerance
//
// solution will be put in \alpha, objective value will be put in obj
internal class Solver {
    //  Keep a reference to the data set
    var problemData : DataSet?
    var outputs: [Double]
    
    //  parameters
    let kernelParams: KernelParameters
    let ϵ : Double
    var costPositive = 0.0
    var costNegative = 0.0
    
    var obj : Double
    var ρ : Double
    var positiveUpperBound : Double
    var negativeUpperBound : Double
    
    var α : [Double]            //  Alpha values
    var αStatus : [AlphaState]  //  alpha status enumerations,sized for the problem
    var gradient : [Double]     //  Gradient vector, one component per data point
    var gradientBar : [Double]  //  Gradient vector, if free variables are treated as 0 slope
    var kernel : Kernel?
    var QDiagonal : [Double] = []

    init (kernelParams: KernelParameters, ϵ : Double) {
        //  Remember the kernel parameters
        self.kernelParams = kernelParams
        self.ϵ = ϵ
        
        //  Initialize variable
        obj = 0.0
        ρ = 0.0
        positiveUpperBound = 0.0
        negativeUpperBound = 0.0
        
        //  Start with empty arrays until we solve
        outputs = []
        α = []
        αStatus = []
        gradient = []
        gradientBar = []
    }
    
    //  Solve a classification problem
    func solveClassification(data: DataSet, display : Bool = false)
    {
        //  Let all routines have access to the data
        problemData = data
        
        //  Allocate the α array for the sub-problem (one per constraint)
        α = [Double](count: data.size, repeatedValue: 0.0)
        
        //  Make the output array -1 or 1 for positive/negative examples
        outputs = []
        for index in 0..<data.size {
            if let output = data.singleOutput(index) {
                outputs.append((output > 0.0) ? 1.0 : -1.0)
            }
        }
        
        //  Get the kernal for the solution
        kernel = SVCKernel(parameters: kernelParams, data: problemData!, outputs: outputs)
        
        //  Initialize the gradients
        gradient = [Double](count: problemData!.size, repeatedValue: -1.0)
        
        //  Solve
        solve()
        
        //  If display is on, put out the interim data
        if (display && costPositive == costNegative) {
            var sum_alpha = 0.0
            for entry in α { sum_alpha += entry }
            let ν = sum_alpha * costPositive / Double(data.size)
            print ("ν = \(ν)")
        }
        
        //  Multiply each α by the expected output (+/-)
        for index in 0..<data.size {
            α[index] *= outputs[index]
        }
    }
    
    func updateAlphaStatus(index: Int)
    {
        let cost = (outputs[index] > 0.0) ? costPositive : costNegative
        if (α[index] >= cost) {
            αStatus[index] = .UpperBound
        }
        else if(α[index] <= 0) {
            αStatus[index] = .LowerBound
        }
        else {
            αStatus[index] = .Free
        }
        
    }
    
    func solve()
    {
        //  Get needed data
        QDiagonal = kernel!.getQDiagonal()
        
        //  Initialize the alpha status
        αStatus = [AlphaState](count: α.count, repeatedValue: .Free)
        for index in 0..<αStatus.count { updateAlphaStatus(index) }
        
        //  Keep the initial gradient settings
        let initialGradient = gradient
        
        //  Initialize the gradient bars
        gradientBar = [Double](count: gradient.count, repeatedValue: 0.0)
        for index in 0..<gradient.count {
            if (αStatus[index] != .LowerBound) {
                let Q_i = kernel!.getQ(index)
                let alpha_i = α[index]
                for j in 0..<gradient.count {
                    gradient[j] += alpha_i * Q_i[j]
                }
                if (αStatus[index] != .UpperBound) {
                    for j in 0..<gradient.count {
                        gradientBar[j] += Q_i[j] * ((outputs[index] > 0.0) ? costPositive : costNegative)
                    }
                 }
            }
        }
        
        //  Optimization step
        var iter = 0
        let max_iter = max(10000000, problemData!.size>Int.max/100 ? Int.max : 100 * problemData!.size)
        
        while(iter < max_iter) {
            let indices = selectWorkingSet()
            if let indexes = indices {
                let i = indexes.i
                let j = indexes.j
                iter += 1
                print("at iteration \(iter), select indices \(i) and \(j)")
                // update α[i] and α[j], handle bounds carefully
                let Q_i = kernel!.getQ(i)
                let Q_j = kernel!.getQ(j)
                let C_i = (outputs[i] > 0.0) ? costPositive : costNegative
                let C_j = (outputs[j] > 0.0) ? costPositive : costNegative
                let oldα_i = α[i]
                let oldα_j = α[j]
                if (outputs[i] != outputs[j]) {
                    var quad_coef = QDiagonal[i] + QDiagonal[j] + 2.0 * Q_i[j]
                    if (quad_coef <= 0) { quad_coef = 1e-12 }
                    let delta = (-gradient[i]-gradient[j])/quad_coef
                    let diff = α[i] - α[j];
                    α[i] += delta
                    α[j] += delta
                    if (diff > 0.0) {
                        if (α[j] < 0.0) {
                            α[j] = 0.0
                            α[i] = diff
                        }
                    }
                    else {
                        if (α[i] < 0.0) {
                            α[i] = 0.0
                            α[j] = -diff
                        }
                    }
                    if (diff > C_i - C_j) {
                        if (α[i] > C_i) {
                            α[i] = C_i
                            α[j] = C_i - diff
                        }
                    }
                    else {
                        if (α[j] > C_j) {
                            α[j] = C_j
                            α[i] = C_j + diff
                        }
                    }
                }
                else {
                    var quad_coef = QDiagonal[i] + QDiagonal[j] - 2.0 * Q_i[j]
                    if (quad_coef <= 0) { quad_coef = 1e-12 }
                    let delta = (gradient[i]-gradient[j])/quad_coef
                    let sum = α[i] + α[j];
                    α[i] -= delta
                    α[j] += delta
                    if (sum > C_i) {
                        if (α[i] > C_i)
                        {
                            α[i] = C_i
                            α[j] = sum - C_i
                        }
                    }
                    else {
                        if (α[j] < 0.0)
                        {
                            α[j] = 0.0
                            α[i] = sum;
                        }
                    }
                    if(sum > C_j) {
                        if (α[j] > C_j) {
                            α[j] = C_j
                            α[i] = sum - C_j
                        }
                    }
                    else {
                        if (α[i] < 0) {
                            α[i] = 0.0
                            α[j] = sum
                        }
                    }
                }
                
                //  Update the gradient
                let delta_α_i = α[i] - oldα_i;
                let delta_α_j = α[j] - oldα_j;
                for k in 0..<gradient.count {
                    gradient[k] += Q_i[k] * delta_α_i + Q_j[k] * delta_α_j;
                }
                
                // update αStatus and gradientBar
                let ui = (αStatus[i] == .UpperBound)
                let uj = (αStatus[j] == .UpperBound)
                updateAlphaStatus(i)
                updateAlphaStatus(j)
                if(ui != (αStatus[i] == .UpperBound)) {
                    let Q_i = kernel!.getQ(i)
                    if (ui) {
                        for k in 0..<gradient.count  {
                            gradientBar[k] -= C_i * Q_i[k]
                        }
                    }
                    else {
                        for k in 0..<gradient.count {
                            gradientBar[k] += C_i * Q_i[k]
                        }
                    }
                }
                
                if(uj != (αStatus[j] == .UpperBound)) {
                    let Q_j = kernel!.getQ(j)
                    if (uj) {
                        for k in 0..<gradient.count {
                            gradientBar[k] -= C_j * Q_j[k]
                        }
                    }
                    else {
                        for k in 0..<gradient.count {
                            gradientBar[k] += C_j * Q_j[k]
                        }
                    }
                }
            }
            else { break }
        }
        
        //  Calculate ρ
        ρ = calculate_ρ()
        
        //  Calculate objective value
        obj = 0.0
        for i in 0..<gradient.count {
            obj += α[i] * (gradient[i] + initialGradient[i])
        }
        obj *= 0.5
        
        positiveUpperBound = costPositive
        negativeUpperBound = costNegative
    }
    
    func selectWorkingSet() ->(i: Int, j: Int)?
    {
        // return i,j such that
        // i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
        // j: minimizes the decrease of obj value
        //    (if quadratic coefficeint <= 0, replace it with tau)
        //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
        var Gmax = -Double.infinity
        var Gmax2 = -Double.infinity
        var Gmax_idx = -1
        var Gmin_idx = -1
        var obj_diff_min = Double.infinity
        
        for t in 0..<gradient.count {
            if(outputs[t] > 0.0) {
                if (αStatus[t] != .UpperBound) {
                    if(-gradient[t] >= Gmax) {
                        Gmax = -gradient[t]
                        Gmax_idx = t
                    }
                }
            }
            else {
                if (αStatus[t] != .LowerBound) {
                    if(gradient[t] >= Gmax) {
                        Gmax = gradient[t]
                        Gmax_idx = t
                    }
                }
            }
        }
        
        let i = Gmax_idx
        var Q_i: [Double] = []
        if(i != -1) {       // NULL Q_i not accessed: Gmax=-INF if i=-1
            Q_i = kernel!.getQ(i)
        }
        
        for j in 0..<gradient.count {
            if (outputs[j] > 0.0) {
                if (αStatus[j] != .LowerBound) {
                    let grad_diff = Gmax + gradient[j]
                    if (gradient[j] >= Gmax2) {Gmax2 = gradient[j]}
                    if (grad_diff > 0.0) {
                        var obj_diff: Double
                        let quad_coef = QDiagonal[i] + QDiagonal[j] - 2.0 * outputs[i] * Q_i[j]
                        if (quad_coef > 0.0) {
                            obj_diff = -(grad_diff * grad_diff) / quad_coef
                        }
                        else {
                            obj_diff = -(grad_diff*grad_diff) / 1e-12
                        }
                        
                        if (obj_diff <= obj_diff_min) {
                            Gmin_idx = j
                            obj_diff_min = obj_diff
                        }
                    }
                }
            }
            else {
                if (αStatus[j] != .UpperBound) {
                    let grad_diff = Gmax - gradient[j];
                    if (-gradient[j] >= Gmax2) { Gmax2 = -gradient[j]}
                    if (grad_diff > 0.0)  {
                        var obj_diff: Double
                        let quad_coef = QDiagonal[i] + QDiagonal[j] + 2.0 * outputs[i] * Q_i[j]
                        if (quad_coef > 0) {
                            obj_diff = -(grad_diff * grad_diff) / quad_coef
                        }
                        else {
                            obj_diff = -(grad_diff * grad_diff) / 1e-12
                        }
                        
                        if (obj_diff <= obj_diff_min)
                        {
                            Gmin_idx=j
                            obj_diff_min = obj_diff
                        }
                    }
                }
            }
        }
        
        if (Gmax+Gmax2 < ϵ) { return nil }
        
        return (i: Gmax_idx, j: Gmin_idx)
    }
    
    func calculate_ρ() -> Double
    {
        var upperBound = Double.infinity
        var lowerBound = -Double.infinity
        var numberFree = 0
        var sumFree = 0.0
        
        for i in 0..<gradient.count {
            let yG = outputs[i] * gradient[i]
            
            if (αStatus[i] == .UpperBound) {
                if(outputs[i] < 0.0) {
                    upperBound = min(upperBound, yG)
                }
                else {
                    lowerBound = max(lowerBound, yG)
                }
            }
            else if (αStatus[i] == .LowerBound) {
                if(outputs[i] > 0.0) {
                    upperBound = min(upperBound, yG)
                }
                else {
                    lowerBound = max(lowerBound, yG)
                }
            }
            else {
                numberFree += 1
                sumFree += yG
            }
        }
        
        var returnValue: Double
        if (numberFree > 0) {
            returnValue = sumFree / Double(numberFree)
        }
        else {
            returnValue = (upperBound + lowerBound) * 0.5
        }
        
        return returnValue
    }
    
    func solveOneClass(data: DataSet, ν: Double, display : Bool = false)
    {
        //  Let all routines have access to the data
        problemData = data
        
        //  Make sure the expected outputs are 1 for single class
        for index in 0..<data.size {
            do {
                try data.setClass(index, newClass: 1)
            }
            catch {
                return    //  return all zeros if error
            }
        }
        
        //  Allocate and initialize the α array for the sub-problem (one per constraint)
        α = []
        let n = Int(ν * Double(data.size))	// # of alpha's at upper bound
        for _ in 0..<n { α.append(1.0) }
        if (n < data.size) {
            α[n] = ν * Double(data.size) - Double(n)
        }
        for _ in (n+1)..<data.size { α.append(0.0) }
        
        //  Get the kernal for the solution
        kernel = OneClassKernel(parameters: kernelParams, data: problemData!)
        
        //  Initialize the gradients
        gradient = [Double](count: problemData!.size, repeatedValue: 0.0)
        
        //  Solve
        solve()
    }
    
    func solveRegression(data: DataSet, p: Double, display : Bool = false)
    {
        //  Let all routines have access to the data
        problemData = data
        
        //  Allocate and initialize the α array for the sub-problem (two per constraint)
        α = [Double](count: data.size*2, repeatedValue: 0.0)
        
        //  Create the initial gradient and the output arrays
        gradient = []
        outputs = []
        for index in 0..<data.size {
            if let output = data.singleOutput(index) {
                gradient.append(p - output)
                outputs.append(1.0)
            }
        }
        for index in 0..<data.size {
            if let output = data.singleOutput(index) {
                gradient.append(p + output)
                outputs.append(-1.0)
            }
        }
        
        //  Get the kernal for the solution
        kernel = SVRKernel(parameters: kernelParams, data: problemData!)
        
        //  Solve the quadratic program
        solve()
        
        var sum_alpha = 0.0
        var new_alpha : [Double] = []
        for i in 0..<data.size {
            let diff = α[i] - α[i+data.size]
            new_alpha.append(diff)
            sum_alpha += fabs(diff);
        }
        α = new_alpha
        if (display) { print("nu = \(sum_alpha/(costPositive * Double(data.size)))") }
   }
}

internal class Solver_ν : Solver
{
    //  Parameters
    var ν: Double
    
    //  Internal variables
    var last_ρ = 1.0
    
    init (kernelParams: KernelParameters, ϵ : Double, ν: Double) {
        self.ν = ν
        
        super.init(kernelParams: kernelParams, ϵ: ϵ)
    }
    
    override func solveClassification(data: DataSet, display: Bool) {
        //  Let all routines have access to the data
        problemData = data
        
        //  Make the output array -1 or 1 for positive/negative examples
        outputs = []
        for index in 0..<data.size {
            if let output = data.singleOutput(index) {
                outputs.append((output > 0.0) ? 1.0 : -1.0)
            }
        }
        
        //  Allocate and initialize the α array for the sub-problem (one per constraint)
        α = []
        var sum_pos = ν * Double(data.size / 2)
        var sum_neg = ν * Double(data.size / 2)
        
        for i in 0..<data.size {
            if(data.classes![i] == +1) {
                let newα = min(1.0, sum_pos)
                α.append(newα)
                sum_pos -= newα
            }
            else {
                let newα = min(1.0, sum_pos)
                α.append(newα)
                sum_neg -= newα
            }
        }
        
        //  Get the kernal for the solution
        kernel = SVCKernel(parameters: kernelParams, data: problemData!, outputs: outputs)
        
        //  Initialize the gradients
        gradient = [Double](count: problemData!.size, repeatedValue: 0.0)

        //  Solve the quadratic program
        solve()
        
        if (display) { print("C = \(1.0/last_ρ)") }
        
        for i in 0..<α.count {
            α[i] *= outputs[i]/last_ρ
        }
        
        ρ /= last_ρ;
        obj /= (last_ρ * last_ρ);
        positiveUpperBound = 1.0 / last_ρ;
        negativeUpperBound = 1.0 / last_ρ;
    }
    
    override func selectWorkingSet() -> (i: Int, j: Int)? {
        // return i,j such that y_i = y_j and
        // i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
        // j: minimizes the decrease of obj value
        //    (if quadratic coefficeint <= 0, replace it with tau)
        //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
        var Gmaxp = -Double.infinity
        var Gmaxp2 = -Double.infinity
        var Gmaxp_idx = -1
        
        var Gmaxn = -Double.infinity
        var Gmaxn2 = -Double.infinity
        var Gmaxn_idx = -1
        
        var Gmin_idx = -1
        var obj_diff_min = Double.infinity
        
        for t in 0..<gradient.count {
            if(outputs[t] > 0.0) {
                if (αStatus[t] != .UpperBound) {
                    if(-gradient[t] >= Gmaxp) {
                        Gmaxp = -gradient[t]
                        Gmaxp_idx = t
                    }
                }
            }
            else {
                if (αStatus[t] != .LowerBound) {
                    if(gradient[t] >= Gmaxn) {
                        Gmaxn = gradient[t]
                        Gmaxn_idx = t
                    }
                }
            }
        }
        var Q_ip: [Double] = []
        if(Gmaxp_idx != -1) {       // NULL Q_ip not accessed: Gmax=-INF if Gmaxp_idx=-1
            Q_ip = kernel!.getQ(Gmaxp_idx)
        }
        var Q_in: [Double] = []
        if(Gmaxn_idx != -1) {       // NULL Q_in not accessed: Gmax=-INF if Gmaxn_idx=-1
            Q_in = kernel!.getQ(Gmaxn_idx)
        }
        
        for j in 0..<gradient.count {
            if(outputs[j] > 0.0) {
                if (αStatus[j] != .LowerBound) {
                    let grad_diff = Gmaxp + gradient[j]
                    if (gradient[j] >= Gmaxp2) { Gmaxp2 = gradient[j] }
                    if (grad_diff > 0.0)  {
                        var obj_diff : Double
                        let quad_coef = QDiagonal[Gmaxp_idx] + QDiagonal[j] - 2.0 * Q_ip[j]
                        if (quad_coef > 0.0) {
                            obj_diff = -(grad_diff * grad_diff) / quad_coef
                        }
                        else {
                            obj_diff = -(grad_diff * grad_diff) / 1e-12
                        }
                        
                        if (obj_diff <= obj_diff_min) {
                            Gmin_idx = j
                            obj_diff_min = obj_diff
                        }
                    }
                }
            }
            else {
                if (αStatus[j] != .UpperBound) {
                    let grad_diff = Gmaxn - gradient[j]
                    if (-gradient[j] >= Gmaxn2) { Gmaxn2 = -gradient[j] }
                    if (grad_diff > 0.0) {
                        var obj_diff : Double
                        let quad_coef = QDiagonal[Gmaxn_idx] + QDiagonal[j] - 2.0 * Q_in[j]
                        if (quad_coef > 0.0) {
                            obj_diff = -(grad_diff * grad_diff) / quad_coef
                        }
                        else {
                            obj_diff = -(grad_diff * grad_diff) / 1e-12
                        }
                        
                        if (obj_diff <= obj_diff_min) {
                            Gmin_idx = j
                            obj_diff_min = obj_diff
                        }
                    }
                }
            }
        }
        
        
        if (max(Gmaxp + Gmaxp2, Gmaxn + Gmaxn2) < ϵ) { return nil }
        
        if(outputs[Gmin_idx] > 0.0) {
            return (i: Gmaxp_idx, j: Gmin_idx)
        }
        else {
            return (i: Gmaxn_idx, j: Gmin_idx)
        }
    }
    
    override func calculate_ρ() -> Double {
        var nr_free1 = 0
        var nr_free2 = 0
        var ub1 = Double.infinity
        var ub2 = Double.infinity
        var lb1 = -Double.infinity
        var lb2 = -Double.infinity
        var sum_free1 = 0.0
        var sum_free2 = 0.0
        
        for i in 0..<gradient.count {
            if(outputs[i] > 0.0) {
                switch (αStatus[i]) {
                case .LowerBound:
                    ub1 = min(ub1, gradient[i])
                    break
                case .UpperBound:
                    lb1 = max(lb1, gradient[i])
                    break
                case .Free:
                    nr_free1 += 1
                    sum_free1 += gradient[i]
                    break
                }
            }
            else {
                switch (αStatus[i]) {
                case .LowerBound:
                    ub2 = min(ub2, gradient[i])
                    break
                case .UpperBound:
                    lb2 = max(lb2, gradient[i])
                    break
                case .Free:
                    nr_free2 += 1
                    sum_free2 += gradient[i]
                    break
                }
            }
        }
        
        
        var r1,r2 : Double
        if (nr_free1 > 0) {
            r1 = sum_free1 / Double(nr_free1)
        }
        else {
            r1 = (ub1+lb1) * 0.5
        }
        
        if(nr_free2 > 0) {
            r2 = sum_free2 / Double(nr_free2)
        }
        else {
            r2 = (ub2+lb2) * 0.5
        }
        
        last_ρ = (r1+r2) * 0.5
        return last_ρ
    }
    
    override func solveRegression(data: DataSet, p: Double, display : Bool = false)
    {
        //  Let all routines have access to the data
        problemData = data
        
        //  Create the initial alpha, gradient and the output arrays
        var sum = costPositive * ν * Double(data.size) * 0.5
        α = []
        gradient = []
        outputs = []
        for index in 0..<data.size {
            let minsum = min(sum, costPositive)
            α.append(minsum)
            sum -= minsum
            
            if let output = data.singleOutput(index) {
                gradient.append(-output)
                outputs.append(1.0)
            }
        }
        for index in 0..<data.size {
            α.append(α[index])
            
            if let output = data.singleOutput(index) {
                gradient.append(output)
                outputs.append(-1.0)
            }
        }
        
        //  Get the kernal for the solution
        kernel = SVRKernel(parameters: kernelParams, data: problemData!)
        
        //  Solve the quadratic program
        solve()
        
        if (display) { print("epsilon = \(last_ρ)") }
        
        var new_alpha : [Double] = []
        for i in 0..<data.size {
            let diff = α[i] - α[i+data.size]
            new_alpha.append(diff)
        }
        α = new_alpha
    }
}