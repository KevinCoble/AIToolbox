//
//  PrincipalComponantAnalysis.swift
//  AIToolbox
//
//  Created by Kevin Coble on 3/22/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation
import Accelerate


enum PCAError: ErrorType {
    case InvalidDimensions
    case ErrorInSVDParameters
    case SVDDidNotConverge
    case PCANotPerformed
    case TransformError
}

///  Class to perform principal component analysis
public class PCA {
    public private(set) var initialDimension : Int
    public private(set) var reducedDimension : Int
    
    public var μ : [Double] = []       //  Mean of the data set used to find basis vectors
    public var eigenValues : [Double] = []     //  Array of all eigenvalues - this will be sized at initialDimension
    public var basisVectors : [Double] = []    //  Matrix (column-major) of the top 'reducedDimension' eigenvectors that for the new basis
    
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
    public func getReducedBasisVectorSet(data: DataSet) throws
    {
        //  Verify we have a valid setup
        if (initialDimension < 2 || reducedDimension < 1 || initialDimension < reducedDimension) {
            throw PCAError.InvalidDimensions
        }
        
        //  Verify the data set matches the initial dimension
        if (data.inputDimension != initialDimension) {
            throw MachineLearningError.DataWrongDimension
        }
        
        //  Make sure we have enough data
        if (data.size < 2) {
            throw MachineLearningError.NotEnoughData
        }
        
        //  Get the mean of the data
        μ = [Double](count: initialDimension, repeatedValue: 0.0)
        for point in 0..<data.size {
            vDSP_vaddD(data.inputs[point], 1, μ, 1, &μ, 1, vDSP_Length(initialDimension))
        }
        var scale = 1.0 / Double(data.size)
        vDSP_vsmulD(μ, 1, &scale, &μ, 1, vDSP_Length(initialDimension))
        
        //  Get the data matrix with a mean of 0 - in column-major format for the LAPACK routines
        var X = [Double](count: initialDimension * data.size, repeatedValue: 0.0)
        var row : [Double] = [Double](count: initialDimension, repeatedValue: 0.0)
        for point in 0..<data.size {
            vDSP_vsubD(μ, 1, data.inputs[point], 1, &row, 1, vDSP_Length(initialDimension))
            for column in 0..<initialDimension {
                X[column * data.size + point] = row[column]
            }
        }
        
        //  Get the SVD decomposition of the X matrix
        let jobZChar = "S" as NSString
        var jobZ : Int8 = Int8(jobZChar.characterAtIndex(0))          //  return min(m,n) rows of Vt
        var m : Int32 = Int32(data.size)
        var n : Int32 = Int32(initialDimension)
        eigenValues = [Double](count: initialDimension, repeatedValue: 0.0)
        var u = [Double](count: data.size * data.size, repeatedValue: 0.0)
        var vTranspose = [Double](count: initialDimension * initialDimension, repeatedValue: 0.0)
        var work : [Double] = [0.0]
        var lwork : Int32 = -1        //  Ask for the best size of the work array
        let iworkSize = 8 * Int(min(m,n))
        var iwork = [Int32](count: iworkSize, repeatedValue: 0)
        var info : Int32 = 0
        dgesdd_(&jobZ, &m, &n, &X, &m, &eigenValues, &u, &m, &vTranspose, &n, &work, &lwork, &iwork, &info)
        if (info != 0 || work[0] < 1) {
            throw PCAError.ErrorInSVDParameters
        }
        lwork = Int32(work[0])
        work = [Double](count: Int(work[0]), repeatedValue: 0.0)
        dgesdd_(&jobZ, &m, &n, &X, &m, &eigenValues, &u, &m, &vTranspose, &n, &work, &lwork, &iwork, &info)
        if (info < 0) {
            throw PCAError.ErrorInSVDParameters
        }
        if (info > 0) {
            throw PCAError.SVDDidNotConverge
        }
        
        //  Extract the new basis vectors - make a row-major matrix for dataset matrix multiplication using vDSP
        basisVectors = [Double](count: reducedDimension * initialDimension, repeatedValue: 0.0)
        for vector in 0..<reducedDimension {
            for column in 0..<initialDimension {
                basisVectors[(vector * reducedDimension) + column] = vTranspose[vector + (column * initialDimension)]
            }
        }
    }
    
    ///  Routine to transform the given dataset into a new dataset using the basis vectors calculated
    public func transformDataSet(data: DataSet) throws ->DataSet
    {
        //  Make sure we have the PCA results to use
        if (basisVectors.count <= 0) { throw PCAError.PCANotPerformed }
        
        //  Make sure the data dimension matches
        if (data.inputDimension != initialDimension) { throw MachineLearningError.DataWrongDimension }
        
        //  Make a new data set with the new dimension
        let result = DataSet(dataType: .Regression, inputDimension: reducedDimension, outputDimension: 1)
        
        //  Convert each data point
        var centered = [Double](count: initialDimension, repeatedValue: 0.0)
        var transformed = [Double](count: reducedDimension, repeatedValue: 0.0)
        for point in 0..<data.size {
            //  Move relative to the mean of the training data
            vDSP_vsubD(data.inputs[point], 1, μ, 1, &centered, 1, vDSP_Length(initialDimension))
            
            //  Convert to the new basis vector
            vDSP_mmulD(basisVectors, 1, centered, 1, &transformed, 1, vDSP_Length(reducedDimension), vDSP_Length(1), vDSP_Length(initialDimension))
            
            //  Add to the new dataset
            do {
                try result.addUnlabeledDataPoint(input: transformed)
            }
            catch {
                throw PCAError.TransformError
            }
        }
        
        //  Return the result
        return result
    }
    
    ///  Routine to write the model result parameters to a property list path at the provided path
    public enum PCAWriteErrors: ErrorType { case failedWriting }
    public func saveToFile(path: String) throws
    {
        //  Create a property list of the PCA model
        var modelDictionary = [String: AnyObject]()
        modelDictionary["initialDimension"] = initialDimension
        modelDictionary["reducedDimension"] = reducedDimension
        modelDictionary["mean"] = μ
        modelDictionary["eigenValues"] = eigenValues
        modelDictionary["basisVectors"] = basisVectors

        //  Convert to a property list (NSDictionary) and write
        let pList = NSDictionary(dictionary: modelDictionary)
        if !pList.writeToFile(path, atomically: false) { throw PCAWriteErrors.failedWriting }
    }
}
