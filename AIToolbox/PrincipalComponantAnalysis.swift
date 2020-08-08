//
//  PrincipalComponantAnalysis.swift
//  AIToolbox
//
//  Created by Kevin Coble on 3/22/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation
import Accelerate


enum PCAError: Error {
    case invalidDimensions
    case errorInSVDParameters
    case svdDidNotConverge
    case pcaNotPerformed
    case transformError
}

///  Class to perform principal component analysis
open class PCA {
    open fileprivate(set) var initialDimension : Int
    open fileprivate(set) var reducedDimension : Int
    
    open var μ : [Double] = []       //  Mean of the data set used to find basis vectors
    open var eigenValues : [Double] = []     //  Array of all eigenvalues - this will be sized at initialDimension
    open var basisVectors : [Double] = []    //  Matrix (column-major) of the top 'reducedDimension' eigenvectors that for the new basis
    
    public init(initialSize: Int, reduceSize: Int)
    {
        initialDimension = initialSize
        reducedDimension = reduceSize
    }
    
    public init?(loadFromFile path: String)
    {
        //  Initialize all the stored properties (Swift requires this, even when returning nil [supposedly fixed in Swift 2.2)
        initialDimension = 0
        reducedDimension = 0
        μ = []
        eigenValues = []
        basisVectors = []
        
        //  Read the property list
        let pList = NSDictionary(contentsOfFile: path)
        if pList == nil { return nil }
        let dictionary : Dictionary = pList! as! Dictionary<String, AnyObject>
        
        //  Get the initial and reduced dimensions from the dictionary
        let initialDimensionValue = dictionary["initialDimension"] as? NSInteger
        if initialDimensionValue == nil { return nil }
        initialDimension = initialDimensionValue!
        let reducedDimensionValue = dictionary["reducedDimension"] as? NSInteger
        if reducedDimensionValue == nil { return nil }
        reducedDimension = reducedDimensionValue!
        
        let meanArray = dictionary["mean"] as? NSArray
        if meanArray == nil { return nil }
        μ = meanArray! as! [Double]
        
        let eigenValueArray = dictionary["eigenValues"] as? NSArray
        if eigenValueArray == nil { return nil }
        eigenValues = eigenValueArray! as! [Double]
        
        let eigenVectorArray = dictionary["basisVectors"] as? NSArray
        if eigenVectorArray == nil { return nil }
        basisVectors = eigenVectorArray! as! [Double]
    }
    
    ///  Routine to get the reduced eigenvector set that is the basis for the reduced dimension subspace
    open func getReducedBasisVectorSet(_ data: MLDataSet) throws
    {
        //  Verify we have a valid setup
        if (initialDimension < 2 || reducedDimension < 1 || initialDimension < reducedDimension) {
            throw PCAError.invalidDimensions
        }
        
        //  Verify the data set matches the initial dimension
        if (data.inputDimension != initialDimension) {
            throw MachineLearningError.dataWrongDimension
        }
        
        //  Make sure we have enough data
        if (data.size < 2) {
            throw MachineLearningError.notEnoughData
        }
        
        //  Get the mean of the data
        μ = [Double](repeating: 0.0, count: initialDimension)
        for point in 0..<data.size {
            let inputs = try data.getInput(point)
            vDSP_vaddD(inputs, 1, μ, 1, &μ, 1, vDSP_Length(initialDimension))
        }
        var scale = 1.0 / Double(data.size)
        vDSP_vsmulD(μ, 1, &scale, &μ, 1, vDSP_Length(initialDimension))
        
        //  Get the data matrix with a mean of 0 - in column-major format for the LAPACK routines
        var X = [Double](repeating: 0.0, count: initialDimension * data.size)
        var row : [Double] = [Double](repeating: 0.0, count: initialDimension)
        for point in 0..<data.size {
            let inputs = try data.getInput(point)
            vDSP_vsubD(μ, 1, inputs, 1, &row, 1, vDSP_Length(initialDimension))
            for column in 0..<initialDimension {
                X[column * data.size + point] = row[column]
            }
        }
        
        //  Get the SVD decomposition of the X matrix
        let jobZChar = "S" as NSString
        var jobZ : Int8 = Int8(jobZChar.character(at: 0))          //  return min(m,n) rows of Vt        var q : __CLPK_integer
        var m : __CLPK_integer = __CLPK_integer(data.size)
        var lda = m
        var ldu = m
        var n : __CLPK_integer = __CLPK_integer(initialDimension)
        var ldvt = n
        eigenValues = [Double](repeating: 0.0, count: initialDimension)
        var u = [Double](repeating: 0.0, count: data.size * data.size)
        var vTranspose = [Double](repeating: 0.0, count: initialDimension * initialDimension)
        var work : [Double] = [0.0]
        var lwork : __CLPK_integer = -1        //  Ask for the best size of the work array
        let iworkSize = 8 * __CLPK_integer(min(m,n))
        var iwork = [__CLPK_integer](repeating: 0, count: Int(iworkSize))
        var info : __CLPK_integer = 0
        dgesdd_(&jobZ, &m, &n, &X, &lda, &eigenValues, &u, &ldu, &vTranspose, &ldvt, &work, &lwork, &iwork, &info)
        if (info != 0 || work[0] < 1) {
            throw PCAError.errorInSVDParameters
        }
        lwork = __CLPK_integer(work[0])
        work = [Double](repeating: 0.0, count: Int(work[0]))
        dgesdd_(&jobZ, &m, &n, &X, &lda, &eigenValues, &u, &ldu, &vTranspose, &ldvt, &work, &lwork, &iwork, &info)
        if (info < 0) {
            throw PCAError.errorInSVDParameters
        }
        if (info > 0) {
            throw PCAError.svdDidNotConverge
        }
        
        //  Extract the new basis vectors - make a row-major matrix for dataset matrix multiplication using vDSP
        basisVectors = [Double](repeating: 0.0, count: reducedDimension * initialDimension)
        for vector in 0..<reducedDimension {
            for column in 0..<initialDimension {
                basisVectors[(vector * initialDimension) + column] = vTranspose[vector + (column * initialDimension)]
            }
        }
    }
    
    ///  Routine to transform the given dataset into a new dataset using the basis vectors calculated
    open func transformDataSet(_ data: MLDataSet) throws ->DataSet
    {
        //  Make sure we have the PCA results to use
        if (basisVectors.count <= 0) { throw PCAError.pcaNotPerformed }
        
        //  Make sure the data dimension matches
        if (data.inputDimension != initialDimension) { throw MachineLearningError.dataWrongDimension }
        
        //  Make a new data set with the new dimension
        let result = DataSet(dataType: .regression, inputDimension: reducedDimension, outputDimension: 1)
        
        //  Convert each data point
        var centered = [Double](repeating: 0.0, count: initialDimension)
        var transformed = [Double](repeating: 0.0, count: reducedDimension)
        for point in 0..<data.size {
            //  Move relative to the mean of the training data
            let inputs = try data.getInput(point)
            vDSP_vsubD(μ, 1, inputs, 1, &centered, 1, vDSP_Length(initialDimension))
            
            //  Convert to the new basis vector
            vDSP_mmulD(basisVectors, 1, centered, 1, &transformed, 1, vDSP_Length(reducedDimension), vDSP_Length(1), vDSP_Length(initialDimension))
            
            //  Add to the new dataset
            do {
                try result.addUnlabeledDataPoint(input: transformed)
            }
            catch {
                throw PCAError.transformError
            }
        }
        
        //  Return the result
        return result
    }
    
    ///  Routine to write the model result parameters to a property list path at the provided path
    public enum PCAWriteErrors: Error { case failedWriting }
    open func saveToFile(_ path: String) throws
    {
        //  Create a property list of the PCA model
        var modelDictionary = [String: AnyObject]()
        modelDictionary["initialDimension"] = initialDimension as AnyObject?
        modelDictionary["reducedDimension"] = reducedDimension as AnyObject?
        modelDictionary["mean"] = μ as AnyObject?
        modelDictionary["eigenValues"] = eigenValues as AnyObject?
        modelDictionary["basisVectors"] = basisVectors as AnyObject?

        //  Convert to a property list (NSDictionary) and write
        let pList = NSDictionary(dictionary: modelDictionary)
        if !pList.write(toFile: path, atomically: false) { throw PCAWriteErrors.failedWriting }
    }
}
