//
//  MLPersistenceProtocol.swift
//  AIToolbox
//
//  Created by Kevin Coble on 7/3/16.
//  Copyright © 2016 Kevin Coble. All rights reserved.
//

import Foundation

public protocol MLPersistence {
    init?(fromDictionary: [String: AnyObject])
    func getPersistenceDictionary() -> [String: AnyObject]
}


extension MLPersistence {
    //  Get an integer array from a source dictionary
    public func getIntArray(_ sourceDictionary: [String: AnyObject], identifier: String) -> [Int]?
    {
        let nsArray = sourceDictionary[identifier] as? NSArray
        if (nsArray == nil)  { return nil }
        var returnArray : [Int] = []
        for item in nsArray! {
            let element = item as? NSInteger
            if (element == nil)  { return nil }
            returnArray.append(element!)
        }
        return returnArray
    }
    
    //  Get an float array from a source dictionary
    public func getFloatArray(_ sourceDictionary: [String: AnyObject], identifier: String) -> [Float]?
    {
        let nsArray = sourceDictionary[identifier] as? NSArray
        if (nsArray == nil)  { return nil }
        var returnArray : [Float] = []
        for item in nsArray! {
            let element = item as? NSNumber
            if (element == nil)  { return nil }
            returnArray.append(element!.floatValue)
        }
        return returnArray
    }
    
    //  Get an double array from a source dictionary
    public func getDoubleArray(_ sourceDictionary: [String: AnyObject], identifier: String) -> [Double]?
    {
        let nsArray = sourceDictionary[identifier] as? NSArray
        if (nsArray == nil)  { return nil }
        var returnArray : [Double] = []
        for item in nsArray! {
            let element = item as? NSNumber
            if (element == nil)  { return nil }
            returnArray.append(element!.doubleValue)
        }
        return returnArray
    }
}
