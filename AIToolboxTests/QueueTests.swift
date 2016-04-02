//
//  QueueTests.swift
//  AIToolbox
//
//  Created by Kevin Coble on 2/20/15.
//  Copyright (c) 2015 Kevin Coble. All rights reserved.
//

import Foundation
import XCTest
import AIToolbox

class QueueTests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }

    func testQueue() {
        //   Create an integer queue
        let queue = Queue<Int>()
        
        //  Verify it starts empty
        XCTAssert(queue.isEmpty, "queue starts empty")
        
        //  Add an item
        queue.enqueue(1)
        XCTAssert(queue.queuedItemCount == 1, "queue has 1 item")
        
        //  Item can be 'peeked'
        XCTAssert(queue.peek() == 1, "queue has 1 item")
        
        //  Add another item
        queue.enqueue(2)
        XCTAssert(queue.queuedItemCount == 2, "queue has 2 items")
        XCTAssert(queue.peek() == 1, "queue puts second item at end")
        
        //  Get an item
        let result = queue.dequeue()
        XCTAssert(result != nil, "queue dequeued 1 item")
        XCTAssert(result! == 1, "queue dequeued first item")
        XCTAssert(queue.queuedItemCount == 1, "queue has 1 item after dequeue")
        XCTAssert(queue.peek() == 2, "queue moved item forward in queue")
        
        //  Get last item
        let secondResult = queue.dequeue()
        XCTAssert(secondResult != nil, "queue dequeued second item")
        XCTAssert(secondResult! == 2, "queue dequeued second item")
        XCTAssert(queue.isEmpty, "queue has no more items after dequeue")
        XCTAssert(queue.peek() == nil, "queue has no more items")
        
        //  No more items
        let thirdResult = queue.dequeue()
        XCTAssert(thirdResult == nil, "queue has no more items at the end")
    }

}
