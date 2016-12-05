import UIKit
import PlaygroundSupport
/*:
 ## Connect-Four style game using Alpha-Beta algorithm
 This playground uses UIKit and the alpha-beta algorithm from [AIToolbox](https://github.com/KevinCoble/AIToolbox) to create a connet-four style game.  The playground shows how to create a class that the alpha-beta algorithm can use to find the best move given the current board position.
 
 The AIToolbox code needed is tucked-away in the **Sources** section of the project navigator, if you want to see that.  The main file shown here has two custom class definitions, one for the view to display the game and handle user input, and one that represents the board.  It is the class that represents the board that is made to conform to the **AlphaBetaNode** protocol, so it can be used with the **AlphaBetaGraph** class of AIToolbox.
 
 The first class defined is the view class for displaying the board.  It derives from UIView.  The relative sizing and the colors of things shown on the board are defined as constants at the top of the class, and can be changed if you want.
 */


class Connect4View : UIView {
    let controlMargin : CGFloat = 0.1       //  Percentage of view area for controls
    let resetWidth : CGFloat = 0.2          //  Percentage of control area for reset
    let pieceMargin : CGFloat = 0.2       //  Percentage of board area for inter-piece spacing
    let pieceColor = [ UIColor.lightGray,   //  Empty color
                       UIColor.red,         //  Player color
                       UIColor.black        //  Computer color
                    ]
    
    override public init(frame: CGRect) {
        board = Connect4Game()
        super.init(frame: frame)
    }
    
/*:
#### Member Variables for Game Play
Member variables for the current board position, whether the computer is thinking (this is done in a background thread so the UI can update), the alpha-beta search depth, and whether the alpha-beta routine should evaluate child nodes concurrently are defined.
     
The **lookAheadDepth** defines the number of turns ahead the alpha-beta algorithm will look.  Since each move usually has the number of possible plays as there are columns (7 in the default case), increasing this value will make the computer think that many times more (again, 7 times more for the default case) before making its move, but the move will likely be higher quality.
*/
    var board : Connect4Game
    var awaitingComputerMove = false
    let lookAheadDepth = 3
    let evaluateConcurrent = true
    
    convenience init(frame: CGRect, game: Connect4Game) {
        self.init(frame: frame)
        board = game
    }
    
    required public init?(coder aDecoder: NSCoder) {
        board = Connect4Game()
        super.init(coder: aDecoder)
    }
/*:
#### Drawing the board
The routine for drawing the board gets the board layout from the **Connect4Game** class, then calculates the sizing parameters of each item before drawing.
     
If the state is currently 'computers move', and the computer is not already thinking, the draw routine will start the background thread that determines the computers move.  This allows drawing to be complete before the CPU becomes busy determing the next move.
 */
 
    override open func draw(_ rect: CGRect) {
        //  Draw the controls rectangle
        let controlHeight = rect.size.height * controlMargin
        var controlRect = rect
        controlRect.size.height = controlHeight
        UIColor.darkGray.setFill()
        var bpath = UIBezierPath(rect: controlRect)
        bpath.fill()
        
        //  Get the attributes for drawing the text
        let paraStyle = NSMutableParagraphStyle()
        paraStyle.lineSpacing = 6.0
        paraStyle.alignment = NSTextAlignment.center
        let resetAttributes = [
            NSForegroundColorAttributeName: UIColor.white,
            NSParagraphStyleAttributeName: paraStyle,
            //NSTextAlignment: textalign,
            NSFontAttributeName: UIFont(name: "Helvetica Neue", size: 24.0)!
            ] as [String : Any]
        let stateAttributes = [
            NSForegroundColorAttributeName: UIColor.blue,
            NSParagraphStyleAttributeName: paraStyle,
            //NSTextAlignment: textalign,
            NSFontAttributeName: UIFont(name: "Helvetica Neue", size: 20.0)!
            ] as [String : Any]

        //  Draw the reset button
        var resetRect = controlRect
        resetRect.origin.y += 50.0 * controlMargin
        resetRect.size.width = controlRect.width * resetWidth
        "Reset".draw(in: resetRect, withAttributes: resetAttributes)
        
        //  Draw the state
        var stateRect = controlRect
        stateRect.size.width = controlRect.width - resetRect.width
        stateRect.origin.y += 100.0 * controlMargin
        board.currentState.rawValue.draw(in: stateRect, withAttributes: stateAttributes)
        
        //  draw the board background
        var boardRect = rect
        boardRect.origin.y += controlHeight
        boardRect.size.height -= controlHeight
        UIColor.cyan.setFill()
        bpath = UIBezierPath(rect: boardRect)
        bpath.fill()
        
        //  Get the size of each piece
        let pieceXMargin = boardRect.width * pieceMargin / CGFloat(Connect4Game.boardColumns)
        let pieceXWidth = (boardRect.width / CGFloat(Connect4Game.boardColumns)) - pieceXMargin
        let pieceYMargin = boardRect.height * pieceMargin / CGFloat(Connect4Game.boardRows)
        let pieceYHeight = (boardRect.height / CGFloat(Connect4Game.boardRows)) - pieceYMargin
        
        //  Draw each piece
        var ypos = boardRect.height - (pieceYHeight + pieceYMargin * 0.5) + controlHeight
        for row in 0..<Connect4Game.boardRows {
            var xpos = pieceXMargin * 0.5
            for column in 0..<Connect4Game.boardColumns {
                let circleRect = CGRect(x: xpos, y: ypos, width: pieceXWidth, height: pieceYHeight)
                let cPath: UIBezierPath = UIBezierPath(ovalIn: circleRect)
                let boardRow = board.board[row]
                pieceColor[boardRow[column].rawValue].setFill()
                cPath.fill()
                xpos += pieceXWidth + pieceXMargin
            }
            ypos -= pieceYHeight + pieceYMargin
        }
        
        //  If the mode is now computers' move, and we haven't started thinking - do so now
        if (board.currentState == .computersMove && !awaitingComputerMove) {
            awaitingComputerMove = true
            let tQueue = DispatchQueue.global(qos: DispatchQoS.QoSClass.default)
            tQueue.async {
                self.getComputersMove()
            }
        }
    }
    
/*:
#### Handling touch events
The only touch event handled is the 'touchesEnded' event.  This is processed like a tap gesture would, but is easier to manage in a playground.
 
The location of tap is determined.  If it is the area for the 'reset' word, a reset is performed.  Otherwise, the column touched is determined and the players move is made into that column.  The board is redrawn (with a setNeedsDisplay call), which will start the determination of the computers response when the drawing is done.
 */
    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        if let touch = touches.first {
            let touchLocation = touch.location(in:self)
            
            //  See if it is in the control area
            var resetRect = bounds
            resetRect.size.height = bounds.height * controlMargin
            resetRect.size.width = bounds.width * resetWidth
            if (touchLocation.y < resetRect.height) {
                if (touchLocation.x < resetRect.width) {
                    awaitingComputerMove = false    //  Stop any thinking
                    board.reset()
                    self.setNeedsDisplay()
                }
                return
            }
            
            let column = Int(touchLocation.x * CGFloat(Connect4Game.boardColumns) / self.bounds.size.width)
            board.playMove(atColumn: column, checkState: true)
            self.setNeedsDisplay()
        }
    }
    
/*:
#### Getting the computers move
The getComputersMove function is called in a background thread to determine the move the computer will make given the current board position.

An instance of the **AlphaBetaNode** class is created, and depending on whether concurrent evaluation has been selected, either the startAlphaBetaConcurrentWithNode or startAlphaBetaWithNode method is called to do the alpha-beta pruning search, using the current board as the start node, and going for the specified depth.  If a move is not determined by the algorithm, a random move is selected.
    
After the move is determiend, the move is played onto the board and the display updated.  As these operations have to happen on the main thread, Grand Central Dispatch is used to do that
*/
    func getComputersMove()
    {
        //  Use the alpha-beta algorithm to find the best move
        let alphaBeta = AlphaBetaGraph()
        var bestMove : AlphaBetaNode!
        if (evaluateConcurrent) {
            bestMove = alphaBeta.startAlphaBetaConcurrentWithNode(board, forDepth: lookAheadDepth)
        }
        else {
            bestMove = alphaBeta.startAlphaBetaWithNode(board, forDepth: lookAheadDepth)
        }
        var column : Int
        if (bestMove != nil) {
            column = (bestMove! as! Connect4Game).lastMove
        }
        else {
            //  Couldn't find a move, go for a random one
            repeat {
                column = Int(arc4random_uniform(UInt32(Connect4Game.boardColumns)))
            } while (board.board[Connect4Game.boardRows-1][column] != .empty)
        }
        
        awaitingComputerMove = false
        DispatchQueue.main.async {
            _ = self.board.playMove(atColumn: column, checkState: true)
            self.setNeedsDisplay()
        }
    }
}

/*:
## Connect4Game Class
The Connect4Game class represents a board position.  It contains constants defining the board layout (number of rows, columns, pieces-in-a-row to win, etc.), the current board position, and the current state (whose move it is, or is the game over).

The class conforms to the **AlphaBetaNode** protocol, so it can be used in the alpha-beta algorithm
*/

class Connect4Game : AlphaBetaNode {
    enum BoardLocation : Int {
        case empty = 0
        case player
        case computer
    }
    
    enum BoardState : String {
        case playersMove = "Your Move"
        case computersMove = "Computer Thinking"
        case playerWon = "You Won!"
        case computerWon = "The Computer Won"
    }
    static let boardRows = 6
    static let boardColumns = 7
    static let winLength = 4       //  number in a row to win
    
    var board: [[BoardLocation]]
    var currentState :BoardState
    var lastMove = 0
    var gameOver = false
    
    init() {
        board = [[]]
        currentState = .playersMove
        reset()
    }
    
    init(fromBoard: Connect4Game) {
        board = fromBoard.board
        currentState = fromBoard.currentState
    }
    
/*:
#### The Reset method
At initialization, and when the 'reset' area is tapped, the reset method will be called on the game board.  The method removes all pieces from the board, and picks who starts the next game.
*/
    func reset() {
        let row = [BoardLocation](repeating: BoardLocation.empty, count: Connect4Game.boardColumns)
        board = [[BoardLocation]](repeating: row, count: Connect4Game.boardRows)
        
        //  If someone had won, start with the other player
        if (currentState == .playerWon) {
            currentState = .computersMove
        }
        else if (currentState == .computerWon) {
            currentState = .playersMove
        }
    }
    
/*:
#### The playMove method
This method modifies the board position by adding a piece in the column indicated.  The type of piece added depends on the current state - whoever's move it is gets the new piece.  The state is updated to the opponents move, and if state checking is on, the board is examined for a winning state.
*/
    func playMove(atColumn: Int, checkState: Bool) -> Bool
    {
        //  Skip if someone has won
        if (currentState == .playerWon || currentState == .computerWon) { return false }
    
        var moveRow = Connect4Game.boardRows - 1     //  Start at the top
        
        //  Check that a move can be made there
        if board[moveRow][atColumn] != .empty { return false }  //  Cant play here
        
        //  Find the lowest empty row
        while (moveRow > 0) {
            if board[moveRow-1][atColumn] != .empty { break }
            moveRow -= 1
        }
        
        //  Set the board position and change the state
        if (currentState == .playersMove) {
            board[moveRow][atColumn] = .player
            currentState = .computersMove
        }
        else {
            board[moveRow][atColumn] = .computer
            currentState = .playersMove
        }
        lastMove = atColumn
        
        //  If we aren't checking the state, we are done
        if (!checkState) { return true }
        
        //  See if anyone won
        var lastLocation = BoardLocation.empty
        var stringLength = 0
        for row in 0..<Connect4Game.boardRows {      //  Check horizontal
            lastLocation = BoardLocation.empty
            stringLength = 0
            for column in 0..<Connect4Game.boardColumns {
                if (board[row][column] == lastLocation && board[row][column] != .empty) {
                    stringLength += 1
                    if (stringLength == Connect4Game.winLength) {
                        if (lastLocation == .player) {currentState = .playerWon} else {currentState = .computerWon}
                    }
                }
                else {
                    lastLocation = board[row][column]
                    stringLength = 1
                }
            }
        }
        for column in 0..<Connect4Game.boardColumns {        //  Check vertical
            lastLocation = BoardLocation.empty
            stringLength = 0
            for row in 0..<Connect4Game.boardRows {
                if (board[row][column] == lastLocation && board[row][column] != .empty) {
                    stringLength += 1
                    if (stringLength == Connect4Game.winLength) {
                        if (lastLocation == .player) {currentState = .playerWon} else {currentState = .computerWon}
                    }
                }
                else {
                    lastLocation = board[row][column]
                    stringLength = 1
                }
            }
        }
        for startRow in 0..<(Connect4Game.boardRows-Connect4Game.winLength+1) {      //  Check diagonal-left
            for startColumn in (Connect4Game.winLength-1)..<Connect4Game.boardColumns {
                lastLocation = board[startRow][startColumn]
                if (lastLocation == .empty) { continue }
                stringLength = 1
                for offset in 1..<Connect4Game.winLength {
                    let row = startRow + offset
                    let column = startColumn - offset
                    if (board[row][column] != lastLocation) { break }
                    stringLength += 1
                    if (stringLength == Connect4Game.winLength) {
                        if (lastLocation == .player) {currentState = .playerWon} else {currentState = .computerWon}
                    }
                }
            }
        }
        for startRow in 0..<(Connect4Game.boardRows-Connect4Game.winLength+1) {      //  Check diagonal-right
            for startColumn in 0..<(Connect4Game.boardColumns-Connect4Game.winLength+1) {
                lastLocation = board[startRow][startColumn]
                if (lastLocation == .empty) { continue }
                stringLength = 1
                for offset in 1..<Connect4Game.winLength {
                    let row = startRow + offset
                    let column = startColumn + offset
                    if (board[row][column] != lastLocation) { break }
                    stringLength += 1
                    if (stringLength == Connect4Game.winLength) {
                        if (lastLocation == .player) {currentState = .playerWon} else {currentState = .computerWon}
                    }
                }
            }
        }
       
        return true //  Move was accepted
    }
    
/*:
 #### The generateMoves method
 This method is one of two required by the **AlphaBetaNode** protocol.  The method returns an array of nodes that represent each of the possible moves that can occur from the current state.  For this game that is easy - just one copy of the board with the move for every column that is not full.
 */
    func generateMoves(_ forMaximizer: Bool) -> [AlphaBetaNode]     //  Get the nodes for each move below this node
    {
        var moves : [Connect4Game] = []
        
        //  If the game is over, no moves are possible
        if (gameOver) { return moves }
        
        //  Add each column that is not empty
        for column in 0..<Connect4Game.boardColumns {
            if board[Connect4Game.boardRows-1][column] == .empty {
                let newBoard = Connect4Game(fromBoard: self)
                newBoard.playMove(atColumn: column, checkState: false)
                moves.append(newBoard)
            }
        }
        
        return moves
    }

/*:
#### The staticEvaluation method
This method is one of two required by the **AlphaBetaNode** protocol.  The method determines the relative 'worth' of the board position, from the perspective of the maximizer.  In this example, the computer is the maximizer - we are trying to determine the computer's move, so we want to maximize its future reward.
     
The method finds each consecutive chain of locations that are the winning length and counts the number of pieces of each type in it.  If there are mixed pieces, then no score is tallied for that set of locations, as a winning play can't be made there.  If a winning sequence is found, it gets 1000 points (or -1000 if it is the player winning).  One piece missing gets 1000 * 0.01, two missing get 1000 * 0.01 * 0.01, etc.
*/
    func staticEvaluation() -> Double                               //  Evaluate the worth of this node
    {
        //  Set the worth of a number of pieces in a row/column/diag without obstruction
        var countWorth = [Double](repeating: 0.0, count: Connect4Game.winLength+1)
        countWorth[Connect4Game.winLength] = 1000.0
        for index in stride(from: (Connect4Game.winLength-1), through: 1, by: -1) {
            countWorth[index] = countWorth[index+1] * 0.01
        }
        
        var value = 0.0
        
        //  Check horizontal
        var computerCount : Int
        var playerCount : Int
        for row in 0..<Connect4Game.boardRows {
            for startColumn in 0..<(Connect4Game.boardColumns-Connect4Game.winLength+1) {
                computerCount = 0
                playerCount = 0
                for offset in 0..<Connect4Game.winLength {
                    let column = startColumn + offset
                    if (board[row][column] == .player) { playerCount += 1 }
                    if (board[row][column] == .computer) { computerCount += 1 }
                }
                if (playerCount == 0) {
                    value += countWorth[computerCount]
                    if (computerCount == Connect4Game.winLength) { gameOver = true }
                }
                if (computerCount == 0) {
                    value -= countWorth[playerCount]
                    if (playerCount == Connect4Game.winLength) { gameOver = true }
                }
            }
        }
        
        //  Check vertical
        for column in 0..<Connect4Game.boardColumns {
            for startRow in 0..<(Connect4Game.boardRows-Connect4Game.winLength+1) {
                computerCount = 0
                playerCount = 0
                for offset in 0..<Connect4Game.winLength {
                    let row = startRow + offset
                    if (board[row][column] == .player) { playerCount += 1 }
                    if (board[row][column] == .computer) { computerCount += 1 }
                }
                if (playerCount == 0) {
                    value += countWorth[computerCount]
                    if (computerCount == Connect4Game.winLength) { gameOver = true }
                }
                if (computerCount == 0) {
                    value -= countWorth[playerCount]
                    if (playerCount == Connect4Game.winLength) { gameOver = true }
                }
            }
        }
        
        //  Check diagonal-left
        for startRow in 0..<(Connect4Game.boardRows-Connect4Game.winLength+1) {
            for startColumn in (Connect4Game.winLength-1)..<Connect4Game.boardColumns {
                computerCount = 0
                playerCount = 0
                for offset in 0..<Connect4Game.winLength {
                    let row = startRow + offset
                    let column = startColumn - offset
                    if (board[row][column] == .player) { playerCount += 1 }
                    if (board[row][column] == .computer) { computerCount += 1 }
                }
                if (playerCount == 0) {
                    value += countWorth[computerCount]
                    if (computerCount == Connect4Game.winLength) { gameOver = true }
                }
                if (computerCount == 0) {
                    value -= countWorth[playerCount]
                    if (playerCount == Connect4Game.winLength) { gameOver = true }
                }
            }
        }
        
        //  Check diagonal-right
        for startRow in 0..<(Connect4Game.boardRows-Connect4Game.winLength+1) {
            for startColumn in 0..<(Connect4Game.boardColumns-Connect4Game.winLength+1) {
                computerCount = 0
                playerCount = 0
                for offset in 0..<Connect4Game.winLength {
                    let row = startRow + offset
                    let column = startColumn + offset
                    if (board[row][column] == .player) { playerCount += 1 }
                    if (board[row][column] == .computer) { computerCount += 1 }
                }
                if (playerCount == 0) {
                    value += countWorth[computerCount]
                    if (computerCount == Connect4Game.winLength) { gameOver = true }
                }
                if (computerCount == 0) {
                    value -= countWorth[playerCount]
                    if (playerCount == Connect4Game.winLength) { gameOver = true }
                }
            }
        }
        
        return value
    }
}

/*:
#### The code that gets run
Now that we have our classes defined, we create a game view and tell the playground to show it.  The view creates the game board to be used.
 */
let gameView = Connect4View(frame: CGRect(x: 0, y: 0, width: 480, height: 440))

PlaygroundPage.current.liveView = gameView
