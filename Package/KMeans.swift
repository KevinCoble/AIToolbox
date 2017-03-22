//
//  KMeans.swift
//  AIToolbox
//
//  Created by Kevin Coble on 3/13/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation
#if os(Linux)
#else
import Accelerate
#endif

enum KMeansError: Error {
    case twoFewPointsForClasses
}


///  Class to perform k-Means grouping of a dataset
open class KMeans {
    var numClasses : Int
    var initWithKPlusPlus = true      //  If false, Forgy initialization (only one point used as centroid per class)
    open fileprivate(set) var centroids : [[Double]]
    
    public init(classes: Int)
    {
        numClasses = classes
        centroids = []
    }
    
    //  Method to classify a dataset
    open func train(_ data: MLClassificationDataSet) throws
    {
        //  If there are not enough points for the classes, throw
        if (data.size < numClasses) { throw KMeansError.twoFewPointsForClasses }
        
        //  If the number of points exactly matches the number of classes, just assign in order
        if (data.size == numClasses) {
            for classIndex in 0..<numClasses { try data.setClass(classIndex, newClass: classIndex) }
            return
        }
        
        //  Set all the classes to -1 to force initial assignment
        for index in 0..<data.size { try data.setClass(index, newClass: -1) }
        
        //  If a kmeans++, set the centroids to the a point that has a randomly chosen using weighted largest distance from all other centroids
        if (initWithKPlusPlus) {
            //  1. Choose one center uniformly at random from among the data points.
            centroids = []
#if os(Linux)
            var pointIndex = random() % data.size
#else
            var pointIndex = Int(arc4random_uniform(UInt32(data.size)))
#endif
            let inputs = try data.getInput(pointIndex)
            centroids.append(inputs)
            try data.setClass(pointIndex, newClass: 0)
            
            while (centroids.count < numClasses) {
                //  2. For each data point x, compute D(x), the distance between x and the nearest center that has already been chosen.
                var distanceArray : [(index: Int, distanceSquared: Double)] = []
                var totalDistance = 0.0
                for point in 0..<data.size {
                    //  Skip points already assigned
                    let currentClass = try data.getClass(point)
                    if (currentClass >= 0) {
                        continue;
                    }
                    //  Get this points distance to the nearest centroid already determined
                    var distanceSquared = 0.0
                    var minDistanceSquared = Double.infinity
                    for centroid in centroids {
                        let inputs = try data.getInput(point)
                        vDSP_distancesqD(inputs, 1, centroid, 1, &distanceSquared, vDSP_Length(data.inputDimension))
                        if (distanceSquared < minDistanceSquared) {
                            minDistanceSquared = distanceSquared
                        }
                    }
                    
                    //  Add to the distance Array
                    distanceArray.append((index: point, distanceSquared: minDistanceSquared))
                    totalDistance += minDistanceSquared
                }
                
                //  3. Choose one new data point at random as a new center, using a weighted probability distribution where a point x is chosen with probability proportional to D(x)2.
                distanceArray.sort(by: {$0.distanceSquared > $1.distanceSquared})
                totalDistance = sqrt(totalDistance)
#if os(Linux)
                var selectionDistance = Double(random()) * totalDistance / Double(RAND_MAX)
#else
                var selectionDistance = Double(arc4random()) * totalDistance / Double(UInt32.max)
#endif
                selectionDistance *= selectionDistance      //  Square to compare against the list
                var totalDistanceToIndex = 0.0
                for distance in distanceArray {
                    pointIndex = distance.index
                    totalDistanceToIndex += distance.distanceSquared
                    if (selectionDistance < totalDistanceToIndex) {break}
                }
                try data.setClass(pointIndex, newClass: centroids.count)
                let inputs = try data.getInput(pointIndex)
                centroids.append(inputs)
                
                //  4. Repeat Steps 2 and 3 until k centers have been chosen.
            }
        }
        
        //  If using Forgy initialization, set the centroids to those points and go to the convergence iterations
        else {
            //  Get a random point for each class
            var initialSet : [Int] = []
            for _ in 0..<numClasses {
                var pointIndex: Int
                repeat {
#if os(Linux)
                    pointIndex = random() % data.size
#else
                    pointIndex = Int(arc4random_uniform(UInt32(data.size)))
#endif
                } while (initialSet.contains(pointIndex))
                initialSet.append(pointIndex)
            }
            
            //  Set the centroids to those point values
            centroids = []
            for classIndex in 0..<numClasses {
                let inputs = try data.getInput(initialSet[classIndex])
                centroids.append(inputs)
            }
        }
        
        //  Continue until we converge
        var changedClass: Bool
        var distanceSquared : Double = 0.0
        repeat {
            //  Assign each point to the nearest mean
            changedClass = false
            for point in 0..<data.size {
                do {
                    let inputs = try data.getInput(point)
                    var newClass = -1
                    var closestDistanceSquared = Double.infinity
                    for testClass in 0..<numClasses {
                        vDSP_distancesqD(inputs, 1, centroids[testClass], 1, &distanceSquared, vDSP_Length(data.inputDimension))
                        if (distanceSquared < closestDistanceSquared) {
                            newClass = testClass
                            closestDistanceSquared = distanceSquared
                        }
                    }
                    let currentClass = try data.getClass(point)
                    if (newClass != currentClass) {
                        try data.setClass(point, newClass: newClass)
                        changedClass = true
                    }
                }
                catch { print("error indexing input array") }
            }
        
            //  Move the centroid of each class to the mean of all the points assigned to the class
            for testClass in 0..<numClasses {
                var count = 0;
                var startLoc = [Double](repeating: 0.0, count: data.inputDimension)
                centroids[testClass] = [Double](repeating: 0.0, count: data.inputDimension)
                for point in 0..<data.size {
                    do {
                        let inputs = try data.getInput(point)
                        let currentClass = try data.getClass(point)
                        if (currentClass == testClass) {
                            vDSP_vaddD(inputs, 1, startLoc, 1, &startLoc, 1, vDSP_Length(data.inputDimension))
                            count += 1
                        }
                    }
                    catch { print("error indexing class array") }
                }
                if (count > 0) {
                    var scale = 1.0 / Double(count)
                    vDSP_vsmulD(startLoc, 1, &scale, &startLoc, 1, vDSP_Length(data.inputDimension))
                    centroids[testClass] = startLoc
                }
                centroids[testClass] = startLoc
            }
        } while (changedClass)
    }
}
