//
//  Queue.swift
//  AIToolbox
//
//  Created by Kevin Coble on 2/20/15.
//  Copyright (c) 2015 Kevin Coble. All rights reserved.
//

import Foundation

class QueueNode<T> {
    let queuedItem: T
    var nextItemInQueue : QueueNode<T>?
    
    init(itemToQueue: T) {
        queuedItem = itemToQueue
    }
}

open class Queue<T> {
    var itemCount: Int = 0
    var queueHead : QueueNode<T>?
    var queueTail : QueueNode<T>?
    
    public init() {}
    
    ///  Computed property to get the number of items queued
    open var queuedItemCount : Int {return itemCount}
    
    ///  Computed property to determine if a queue is empty
    open var isEmpty: Bool {return itemCount == 0}
    
    ///  Method to add an item to the queue
    open func enqueue(_ itemToQueue: T) {
        let newNode = QueueNode<T>(itemToQueue: itemToQueue)
        if (itemCount == 0) {
            //  First item in becomes both head and tail
            queueHead = newNode
            queueTail = newNode
        } else {
            //  All other items become the new tail
            queueTail?.nextItemInQueue = newNode
            queueTail = newNode
        }
        itemCount += 1
    }

    ///  Method to get the item at the front of the queue.  The item is removed from the queue
    open func dequeue() -> T? {
        //  If no items, return nil
        if (itemCount == 0) {return nil}
        
        //  Remove the head node
        let headNode = queueHead
        queueHead = headNode?.nextItemInQueue

        //  Lower the count
        itemCount -= 1
        if (itemCount == 0) { queueTail = nil}
        
        return headNode!.queuedItem
    }
    
    ///  Method to examine the next item in the queue, without removing it
    open func peek() -> T? {
        //  If no items, return nil
        if (itemCount == 0) {return nil}
        
        return queueHead?.queuedItem
    }
}
