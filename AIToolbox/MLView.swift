import Cocoa


public enum MLViewError: Error {
    case dataSetNotRegression
    case dataSetNotClassification
    case inputVectorNotOfCorrectSize
    case inputIndexOutsideOfRange
    case outputIndexOutsideOfRange
    case allClassificationLabelsNotCovered
}

public enum MLViewAxisSource {
    case dataInput
    case dataOutput
    case classLabel     //  Ignores index
}

public enum MLPlotSymbolShape {
    case circle
    case rectangle
    case plus
    case minus
}

public enum MLViewLegendLocation {
    case upperLeft
    case upperRight
    case lowerLeft
    case lowerRight
    case custom
}

///  Class for symbols showing locations on the plot
open class MLPlotSymbol {
    
    //  Symbol information
    open var symbolColor: NSColor
    open var symbolSize: CGFloat = 7.0
    open var symbolShape: MLPlotSymbolShape = .circle
    
    public init(color: NSColor) {
        symbolColor = color
    }
    
    public convenience init(color: NSColor, symbolShape: MLPlotSymbolShape, symbolSize: CGFloat) {
        self.init(color: color)
        
        self.symbolShape = symbolShape
        self.symbolSize = symbolSize
    }
    
    open func drawAt(_ point: CGPoint) {
        //  Set the color
        symbolColor.set()
        
        switch (symbolShape) {
        case .circle:
            let circleRect = NSMakeRect(point.x - (symbolSize * 0.5), point.y - (symbolSize * 0.5), symbolSize, symbolSize)
            let cPath: NSBezierPath = NSBezierPath(ovalIn: circleRect)
            cPath.fill()
        case .rectangle:
            let rect = NSMakeRect(point.x - (symbolSize * 0.5), point.y - (symbolSize * 0.5), symbolSize, symbolSize)
            NSRectFill(rect)
        case .plus:
            let hrect = NSMakeRect(point.x - (symbolSize * 0.5), point.y - (symbolSize * 0.1), symbolSize, symbolSize * 0.2)
            NSRectFill(hrect)
            let vrect = NSMakeRect(point.x - (symbolSize * 0.1), point.y - (symbolSize * 0.5), symbolSize * 0.2, symbolSize)
            NSRectFill(vrect)
        case .minus:
            let rect = NSMakeRect(point.x - (symbolSize * 0.5), point.y - (symbolSize * 0.1), symbolSize, symbolSize * 0.2)
            NSRectFill(rect)
        }
        
    }
}

public protocol MLViewItem {
    func setColor(_ color: NSColor)       //  Sets the default color for the item
    func setScale(_ scale: (minX: Double, maxX: Double, minY: Double, maxY: Double))      //  Sets the scale to the provided factors, or the item can calculate it's own
    func draw(_ bounds: CGRect)
    func getScale() -> (minX: Double, maxX: Double, minY: Double, maxY: Double)?     //  Return the scale factors used by the item
    func setInputVector(_ vector: [Double]) throws       //  Set the input vector used to get values. Plot variable index (if used) is ignored
    func setXAxisSource(_ source: MLViewAxisSource, index: Int) throws
    func setYAxisSource(_ source: MLViewAxisSource, index: Int) throws
}

///  Class for drawing a regression style data set onto a plot
open class MLViewRegressionDataSet: MLViewItem {
    //  DataSet to be drawn
    let dataset : MLRegressionDataSet

    //  Axis data source
    var sourceTypeXAxis = MLViewAxisSource.dataInput
    var sourceIndexXAxis = 0
    var sourceTypeYAxis = MLViewAxisSource.dataOutput
    var sourceIndexYAxis = 0
    
    //  Scale parameters
    open var scaleToData = true
    open var roundScales = true

    //  Symbol information
    open var symbol = MLPlotSymbol(color: NSColor.green)
    
    //  Axis scaling ranges
    var minX = 0.0
    var maxX = 100.0
    var minY = 0.0
    var maxY = 100.0
    
    public init(dataset: MLRegressionDataSet) throws {
        if (dataset.dataType != .regression) {throw MLViewError.dataSetNotRegression}
        self.dataset = dataset
    }
    
    public convenience init(dataset: DataSet, color: NSColor) throws {
        try self.init(dataset: dataset)
        
        setColor(color)
    }
    
    public convenience init(dataset: DataSet, symbol: MLPlotSymbol) throws {
        try self.init(dataset: dataset)
        
        self.symbol = symbol
    }
    
    open func scaleToData(_ calculateScale : Bool) {
        scaleToData = calculateScale
    }
    
    open func setColor(_ color: NSColor)       //  Sets the default color for the item
    {
        symbol.symbolColor = color
    }
    
    open func setScale(_ scale: (minX: Double, maxX: Double, minY: Double, maxY: Double)) {
        //  Store the scale to be used
        minX = scale.minX
        maxX = scale.maxX
        minY = scale.minY
        maxY = scale.maxY
    }
    
    open func draw(_ bounds: CGRect) {
        //  Get the scaling factors
        let scaleFactorX = CGFloat(1.0 / (maxX - minX))
        let scaleFactorY = CGFloat(1.0 / (maxY - minY))
        
        //  Iterate through each point
        do {
            for point in 0..<dataset.size {
                //  Get the source of the X axis value
                var x_source : [Double]
                if (sourceTypeXAxis == .dataInput) {
                    x_source = try dataset.getInput(point)
                }
                else {
                    x_source = try dataset.getOutput(point)
                }
                //  Get the source of the Y axis value
                var y_source : [Double]
                if (sourceTypeYAxis == .dataInput) {
                    y_source = try dataset.getInput(point)
                }
                else {
                    y_source = try dataset.getOutput(point)
                }
                //  Calculate the plot position and draw
                let x = (CGFloat(x_source[sourceIndexXAxis] - minX) * scaleFactorX) * bounds.width + bounds.origin.x
                let y = (CGFloat(y_source[sourceIndexYAxis] - minY) * scaleFactorY) * bounds.height + bounds.origin.y
                symbol.drawAt(CGPoint(x: x, y: y))
            }
        }
        catch {
            //  Skip this dataset
        }
    }
    
    open func getScale() -> (minX: Double, maxX: Double, minY: Double, maxY: Double)? {
        //  If we are scaling to the data, determine the scale factors now
        if (scaleToData) {
            //  Get the source of the X axis range
            var x_range : [(minimum: Double, maximum: Double)]
            if (sourceTypeXAxis == .dataInput) {
                x_range = dataset.getInputRange()
            }
            else {
                x_range = dataset.getOutputRange()
            }
            
            //  Get the x axis range
            if (roundScales) {
                let x_scale = MLView.roundScale(x_range[sourceIndexXAxis].minimum, max: x_range[sourceIndexXAxis].maximum)
                minX = x_scale.min
                maxX = x_scale.max
            }
            else {
                minY = x_range[sourceIndexXAxis].minimum
                maxY = x_range[sourceIndexXAxis].maximum
            }
            
            //  Get the source of the Y axis range
            var y_range : [(minimum: Double, maximum: Double)]
            if (sourceTypeYAxis == .dataInput) {
                y_range = dataset.getInputRange()
            }
            else {
                y_range = dataset.getOutputRange()
            }
            
            //  Get the y axis range
            if (roundScales) {
                let y_scale = MLView.roundScale(y_range[sourceIndexYAxis].minimum, max: y_range[sourceIndexYAxis].maximum)
                minY = y_scale.min
                maxY = y_scale.max
            }
            else {
                minY = y_range[sourceIndexYAxis].minimum
                maxY = y_range[sourceIndexYAxis].maximum
            }
        }
        
        return (minX: minX, maxX: maxX, minY: minY, maxY: maxY)
    }
    
    open func setInputVector(_ vector: [Double]) throws {
        //  Not needed for a data set
    }
    
    open func setXAxisSource(_ source: MLViewAxisSource, index: Int) throws {
        switch (source) {
        case .dataInput:
            if (index > dataset.inputDimension) { throw MLViewError.inputIndexOutsideOfRange }
        case .dataOutput:
            if (dataset.dataType != .regression) { throw MLViewError.dataSetNotRegression }
            if (index > dataset.outputDimension) { throw MLViewError.outputIndexOutsideOfRange }
        case .classLabel:     //  Ignores index
            throw MLViewError.dataSetNotClassification
        }
        sourceTypeXAxis = source
        sourceIndexXAxis = index
    }
    open func setYAxisSource(_ source: MLViewAxisSource, index: Int) throws {
        switch (source) {
        case .dataInput:
            if (index > dataset.inputDimension) { throw MLViewError.inputIndexOutsideOfRange }
        case .dataOutput:
            if (dataset.dataType != .regression) { throw MLViewError.dataSetNotRegression }
            if (index > dataset.outputDimension) { throw MLViewError.outputIndexOutsideOfRange }
        case .classLabel:     //  Ignores index
            throw MLViewError.dataSetNotClassification
        }
        sourceTypeYAxis = source
        sourceIndexYAxis = index
    }
}


///  Class for drawing a classification style data set onto a plot
open class MLViewClassificationDataSet: MLViewItem {
    //  DataSet to be drawn
    var dataset : MLClassificationDataSet
    
    //  Axis data source
    var sourceTypeXAxis = MLViewAxisSource.dataInput
    var sourceIndexXAxis = 0
    var sourceTypeYAxis = MLViewAxisSource.dataInput
    var sourceIndexYAxis = 1
    
    //  Scale parameters
    open var scaleToData = true
    open var roundScales = true
    
    //  Symbol information
    open var symbols : [MLPlotSymbol]
    
    //  Axis scaling ranges
    var minX = 0.0
    var maxX = 100.0
    var minY = 0.0
    var maxY = 100.0
    
    public init(dataset: MLClassificationDataSet) throws {
        if (dataset.dataType != .classification) {throw MLViewError.dataSetNotClassification}
        self.dataset = dataset
        
        //  Get the classes
        let optionalData = try dataset.groupClasses()
        self.dataset.optionalData = optionalData
        
        //  Set a symbol for each of them
        symbols = []
        let colors = [
            NSColor.green,
            NSColor.red,
            NSColor.blue,
            NSColor.cyan,
            NSColor.magenta,
            NSColor.yellow,
            NSColor.gray,
            NSColor.black
        ]
        let shapes = [
            MLPlotSymbolShape.circle,
            MLPlotSymbolShape.rectangle,
            MLPlotSymbolShape.plus,
            MLPlotSymbolShape.minus
        ]
        var colorIndex = 0
        var shapeIndex = 0
        var size : CGFloat = 6.0
        for _ in 0..<optionalData.numClasses {
            symbols.append(MLPlotSymbol(color: colors[colorIndex], symbolShape: shapes[shapeIndex], symbolSize: size))
            colorIndex += 1
            if (colorIndex >= colors.count) {
                colorIndex = 0
                shapeIndex += 1
                if (shapeIndex >= shapes.count) {
                    shapeIndex = 0
                    size += 2.0
                }
            }
        }
    }
    
    public convenience init(dataset: DataSet, symbols: [MLPlotSymbol]) throws {
        try self.init(dataset: dataset)
        
        let optionalData = try dataset.groupClasses()
        dataset.optionalData = optionalData
        if (symbols.count < optionalData.numClasses) {throw MLViewError.allClassificationLabelsNotCovered}
        
        self.symbols = symbols
    }
    
    //  Sets ALL symbols to the specified color
    open func setColor(_ color: NSColor)       //  Sets the default color for the item
    {
        for symbol in symbols {
            symbol.symbolColor = color
        }
    }
    
    open func scaleToData(_ calculateScale : Bool) {
        scaleToData = calculateScale
    }
    
    open func setScale(_ scale: (minX: Double, maxX: Double, minY: Double, maxY: Double)) {
        //  Store the scale to be used
        minX = scale.minX
        maxX = scale.maxX
        minY = scale.minY
        maxY = scale.maxY
    }
    
    open func draw(_ bounds: CGRect) {
        //  Get the classes
        do {
            let optionalData = try dataset.groupClasses()
            dataset.optionalData = optionalData
        }
        catch {
            //  Error getting class information for data set, return
            return
        }
        let optionalData = dataset.optionalData as! ClassificationData

        //  Get the scaling factors
        let scaleFactorX = CGFloat(1.0 / (maxX - minX))
        let scaleFactorY = CGFloat(1.0 / (maxY - minY))
        
        //  Iterate through each point
        do {
            for point in 0..<dataset.size {
                //  Get the source of the X axis value
                var x_source : [Double]
                if (sourceTypeXAxis == .dataInput) {
                    x_source = try dataset.getInput(point)
                }
                else {
                    let pointClass = try dataset.getClass(point)
                    x_source = [Double(pointClass)]
                }
                //  Get the source of the Y axis value
                var y_source : [Double]
                if (sourceTypeYAxis == .dataInput) {
                    y_source = try dataset.getInput(point)
                }
                else {
                    let pointClass = try dataset.getClass(point)
                    y_source = [Double(pointClass)]
                }
                //  Calculate the plot position
                let x = (CGFloat(x_source[sourceIndexXAxis] - minX) * scaleFactorX) * bounds.width + bounds.origin.x
                let y = (CGFloat(y_source[sourceIndexYAxis] - minY) * scaleFactorY) * bounds.height + bounds.origin.y
                //  Get the label index
                do {
                    let label = try dataset.getClass(point)
                    var labelIndex = 0
                    for i in 0..<optionalData.numClasses {
                        if (label == optionalData.foundLabels[i]) {
                            labelIndex = i
                            break
                        }
                    }
                    symbols[labelIndex].drawAt(CGPoint(x: x, y: y))
                }
                catch {
                    //  Skip this point if an error getting the label
                }
            }
        }
        catch {
            //  Skip this dataset
        }
    }
    
    open func getScale() -> (minX: Double, maxX: Double, minY: Double, maxY: Double)? {
        //  If we are scaling to the data, determine the scale factors now
        if (scaleToData) {
            //  Get the source of the X axis range
            var x_range : [(minimum: Double, maximum: Double)]
            if (sourceTypeXAxis == .dataInput) {
                x_range = dataset.getInputRange()
            }
            else {
                do {
                    let classificationData = try dataset.groupClasses()
                    x_range = [(minimum: 0.0, maximum: Double(classificationData.numClasses-1))]
                }
                catch {
                    //  Error getting classification data
                    x_range = [(minimum: 0.0, maximum: 1.0)]
                }
            }
            
            //  Get the x axis range
            if (roundScales) {
                let x_scale = MLView.roundScale(x_range[sourceIndexXAxis].minimum, max: x_range[sourceIndexXAxis].maximum)
                minX = x_scale.min
                maxX = x_scale.max
            }
            else {
                minY = x_range[sourceIndexXAxis].minimum
                maxY = x_range[sourceIndexXAxis].maximum
            }
            
            //  Get the source of the Y axis range
            var y_range : [(minimum: Double, maximum: Double)]
            if (sourceTypeYAxis == .dataInput) {
                y_range = dataset.getInputRange()
            }
            else {
                do {
                    let classificationData = try dataset.groupClasses()
                    y_range = [(minimum: 0.0, maximum: Double(classificationData.numClasses-1))]
                }
                catch {
                    //  Error getting classification data
                    y_range = [(minimum: 0.0, maximum: 1.0)]
                }
            }
            
            //  Get the y axis range
            if (roundScales) {
                let y_scale = MLView.roundScale(y_range[sourceIndexYAxis].minimum, max: y_range[sourceIndexYAxis].maximum)
                minY = y_scale.min
                maxY = y_scale.max
            }
            else {
                minY = y_range[sourceIndexYAxis].minimum
                maxY = y_range[sourceIndexYAxis].maximum
            }
        }
        
        return (minX: minX, maxX: maxX, minY: minY, maxY: maxY)
    }
    
    open func setInputVector(_ vector: [Double]) throws {
        //  Not needed for a data set
    }
    
    open func setXAxisSource(_ source: MLViewAxisSource, index: Int) throws {
        switch (source) {
        case .dataInput:
            if (index > dataset.inputDimension) { throw MLViewError.inputIndexOutsideOfRange }
        case .dataOutput:
            throw MLViewError.dataSetNotRegression
        case .classLabel:     //  Ignores index
            if (index > 0) { throw MLViewError.outputIndexOutsideOfRange }
        }
        sourceTypeXAxis = source
        sourceIndexXAxis = index
    }
    open func setYAxisSource(_ source: MLViewAxisSource, index: Int) throws {
        switch (source) {
        case .dataInput:
            if (index > dataset.inputDimension) { throw MLViewError.inputIndexOutsideOfRange }
        case .dataOutput:
            throw MLViewError.dataSetNotRegression
        case .classLabel:     //  Ignores index
            if (index > 0) { throw MLViewError.outputIndexOutsideOfRange }
        }
        sourceTypeYAxis = source
        sourceIndexYAxis = index
    }
}

///  Class for drawing a regression line
open class MLViewRegressionLine: MLViewItem {
    //  Regression line to be drawn
    let regressor : Regressor
    
    //  Axis data source
    var sourceTypeXAxis = MLViewAxisSource.dataInput
    var sourceIndexXAxis = 0
    var sourceTypeYAxis = MLViewAxisSource.dataOutput
    var sourceIndexYAxis = 0
    var inputVector : [Double] = []
    
    //  Line information
    open var lineColor = NSColor.red
    open var lineThickness : CGFloat = 1.0
    
    //  Axis scaling ranges
    var minX = 0.0
    var maxX = 100.0
    var minY = 0.0
    var maxY = 100.0
    
    public init(regressor: Regressor) {
        self.regressor = regressor
        
        inputVector = [Double](repeating: 0.0, count: regressor.getInputDimension())
    }
    
    public convenience init(regressor: Regressor, color: NSColor) {
        self.init(regressor: regressor)
        
        setColor(color)
    }
    
    open func setColor(_ color: NSColor)       //  Sets the default color for the item
    {
        lineColor = color
    }
    
    open func setScale(_ scale: (minX: Double, maxX: Double, minY: Double, maxY: Double)) {
        //  Store the scale
        minX = scale.minX
        maxX = scale.maxX
        minY = scale.minY
        maxY = scale.maxY
    }
    
    open func draw(_ bounds: CGRect) {
        //  Save the current state
        NSGraphicsContext.saveGraphicsState()
        
        //  Set a clip region - the lines aren't usually with the bounds
        NSRectClip(bounds)
        
        //  Get the scaling factors
        let scaleFactorX = CGFloat(1.0 / (maxX - minX))
        let scaleFactorY = CGFloat(1.0 / (maxY - minY))
        
        //  Set the color
        lineColor.setStroke()
        
        //  Get the pixel granularity
        let pixelX = (maxX - minX) / Double(bounds.width)
        let path = NSBezierPath()
        path.lineWidth = lineThickness

        do {
            //  Get the first pixel
            var xIterator = minX - (pixelX * 0.5)
            var inputs = inputVector
            if (sourceTypeXAxis == .dataInput) {
                inputs[sourceIndexXAxis] = xIterator
            }
            let outputs = try regressor.predictOne(inputs)
            //  Get the X axis value
            var xValue : Double
            if (sourceTypeXAxis == .dataInput) {
                xValue = inputs[sourceIndexXAxis]
            }
            else {
                xValue = outputs[sourceIndexXAxis]
            }
            //  Get the Y axis value
            var yValue : Double
            if (sourceTypeYAxis == .dataInput) {
                yValue = inputs[sourceIndexYAxis]
            }
            else {
                yValue = outputs[sourceIndexYAxis]
            }
            //  Get the point coordinates
            let x = (CGFloat(xValue - minX) * scaleFactorX) * bounds.width + bounds.origin.x
            let y = (CGFloat(yValue - minY) * scaleFactorY) * bounds.height + bounds.origin.y
            path.move(to: CGPoint(x: x, y: y))

            //  Iterate through each pixel
            while (xIterator < maxX) {
                inputs = inputVector
                if (sourceTypeXAxis == .dataInput) {
                    inputs[sourceIndexXAxis] = xIterator
                }
                let outputs = try regressor.predictOne(inputs)
                //  Get the X axis value
                if (sourceTypeXAxis == .dataInput) {
                    xValue = inputs[sourceIndexXAxis]
                }
                else {
                    xValue = outputs[sourceIndexXAxis]
                }
                //  Get the Y axis value
                if (sourceTypeYAxis == .dataInput) {
                    yValue = inputs[sourceIndexYAxis]
                }
                else {
                    yValue = outputs[sourceIndexYAxis]
                }
                //  Get the point coordinates
                let x = (CGFloat(xValue - minX) * scaleFactorX) * bounds.width + bounds.origin.x
                let y = (CGFloat(yValue - minY) * scaleFactorY) * bounds.height + bounds.origin.y
                xIterator += pixelX
                path.line(to: CGPoint(x: x, y: y))
            }

            //  Draw the line
            path.stroke()
        }
        catch {
            //  Skip this regressor
        }
        
        //  Restore the current state
        NSGraphicsContext.restoreGraphicsState()
    }
    
    open func getScale() -> (minX: Double, maxX: Double, minY: Double, maxY: Double)? {
        return nil
    }
    
    open func setInputVector(_ vector: [Double]) throws {
        if (regressor.getInputDimension() > vector.count) { throw MLViewError.inputVectorNotOfCorrectSize}
        inputVector = vector
    }
    
    open func setXAxisSource(_ source: MLViewAxisSource, index: Int) throws {
        switch (source) {
        case .dataInput:
            if (index > regressor.getInputDimension()) { throw MLViewError.inputIndexOutsideOfRange }
        case .dataOutput:
            if (index > regressor.getOutputDimension()) { throw MLViewError.outputIndexOutsideOfRange }
        case .classLabel:     //  Ignores index
            throw MLViewError.dataSetNotClassification
        }
        sourceTypeXAxis = source
        sourceIndexXAxis = index
    }
    open func setYAxisSource(_ source: MLViewAxisSource, index: Int) throws {
        switch (source) {
        case .dataInput:
            if (index > regressor.getInputDimension()) { throw MLViewError.inputIndexOutsideOfRange }
        case .dataOutput:
            if (index > regressor.getOutputDimension()) { throw MLViewError.outputIndexOutsideOfRange }
        case .classLabel:     //  Ignores index
            throw MLViewError.dataSetNotClassification
        }
        sourceTypeYAxis = source
        sourceIndexYAxis = index
    }
}


///  Class for drawing a classification area onto a plot
open class MLViewClassificationArea: MLViewItem {
    //  Classifier to be drawn
    let classifier : Classifier
    
    //  Axis data source
    var sourceTypeXAxis = MLViewAxisSource.dataInput
    var sourceIndexXAxis = 0
    var sourceTypeYAxis = MLViewAxisSource.dataInput
    var sourceIndexYAxis = 1
    var inputVector : [Double] = []
    
    //  Label color information
    open var unknownColor = NSColor.white
    open var colors : [NSColor]
    open var granularity = 4      ///  Size of 'pixels' area is drawn in
    
    //  Axis scaling ranges
    var minX = 0.0
    var maxX = 100.0
    var minY = 0.0
    var maxY = 100.0
    
    public init(classifier: Classifier) {
        self.classifier = classifier
        
        let startColors = [
            NSColor.green,
            NSColor.red,
            NSColor.blue,
            NSColor.cyan,
            NSColor.magenta,
            NSColor.yellow,
            NSColor.gray,
            NSColor.black
        ]
        let numClasses = classifier.getNumberOfClasses()
        colors = []
        var colorIndex = 0
        var fadeValue : CGFloat = 0.5
        for _ in 0..<numClasses {
            let red = (1.0 - startColors[colorIndex].redComponent) * fadeValue + startColors[colorIndex].redComponent
            let green = (1.0 - startColors[colorIndex].greenComponent) * fadeValue + startColors[colorIndex].greenComponent
            let blue = (1.0 - startColors[colorIndex].blueComponent) * fadeValue + startColors[colorIndex].blueComponent
            let color = NSColor(red: red, green: green, blue: blue, alpha: 1.0)
            colors.append(color)
            colorIndex += 1
            if (colorIndex >= startColors.count) {
                colorIndex = 0
                fadeValue += (1.0 - fadeValue) * 0.5
            }
        }
        
        inputVector = [Double](repeating: 0.0, count: classifier.getInputDimension())
    }
    
    //  Convenience constructor to create plot object and set colors for the classification labels
    public convenience init(classifier: Classifier, colors: [NSColor]) {
        self.init(classifier: classifier)
        
        self.colors = colors
    }
    
    open func setColor(_ color: NSColor)       //  Sets the default color for the item
    {
        unknownColor = color
    }
    
    open func setScale(_ scale: (minX: Double, maxX: Double, minY: Double, maxY: Double)) {
        //  Store the scale
        minX = scale.minX
        maxX = scale.maxX
        minY = scale.minY
        maxY = scale.maxY
    }
    
    open func draw(_ bounds: CGRect) {
        //  draw the 'other' color
        unknownColor.setFill()
        NSRectFill(bounds)
        
        //  Get the scaling factors
        let scaleFactorX = CGFloat(1.0 / (maxX - minX))
        let scaleFactorY = CGFloat(1.0 / (maxY - minY))
        
        //  Get the pixel granularity
        var grainWidth : CGFloat
        var grainHeight : CGFloat
        if (granularity == 0) {
            grainWidth = 0.5
            grainHeight = 0.5
        }
        else {
            grainWidth = CGFloat(granularity)
            grainHeight = CGFloat(granularity)
        }
        let pixelX = (maxX - minX) * Double(grainWidth) / Double(bounds.width)
        let pixelY = (maxY - minY) * Double(grainWidth)  / Double(bounds.height)
        
        do {
            //  Get the first pixel
            var xIterator = minX
            
            //  Iterate through each pixel
            while (xIterator < maxX) {
                var yIterator = minY
                
                //  Iterate through each pixel
                while (yIterator < maxY) {
                    //  Get the rectangle start coordinates
                    let x = (CGFloat(xIterator - minX) * scaleFactorX) * bounds.width + bounds.origin.x
                    let y = (CGFloat(yIterator - minY) * scaleFactorY) * bounds.height + bounds.origin.y
                    
                    //  Get the rectangle
                    let rect = CGRect(x: x, y: y, width: grainWidth, height: grainHeight)
                    
                    //  Get the class
                    var inputs = inputVector
                    if (sourceTypeXAxis == .dataInput) {
                        inputs[sourceIndexXAxis] = xIterator + (pixelX * 0.5)
                    }
                    if (sourceTypeYAxis == .dataInput) {
                        inputs[sourceIndexYAxis] = yIterator + (pixelY * 0.5)
                    }
                    let label = try classifier.classifyOne(inputs)
                    
                    //  Get the label index
//!!  May need to look into adding label list to classifier
//                    var labelIndex = 0
//                    for i in 0..<optionalData.numClasses {
//                        if (label == optionalData.foundLabels[i]) {
//                            labelIndex = i
//                            break
//                        }
//                    }
                    
                    //  Set the color
                    if (label >= 0 && label < colors.count) {
                        colors[label].set()
                        
                        //  Fill the rectangle
                        let bp = NSBezierPath(rect: rect)
                        bp.fill()
                        bp.stroke()
                    }
                    
                    yIterator += pixelY
                }
                xIterator += pixelX
            }
        }
        catch {
            //  Skip this classifier
        }
    }
    
    open func getScale() -> (minX: Double, maxX: Double, minY: Double, maxY: Double)? {
        return nil
    }
    
    open func setInputVector(_ vector: [Double]) throws {
        if (classifier.getInputDimension() > vector.count) { throw MLViewError.inputVectorNotOfCorrectSize}
        inputVector = vector
    }
    
    open func setXAxisSource(_ source: MLViewAxisSource, index: Int) throws {
        switch (source) {
        case .dataInput:
            if (index > classifier.getInputDimension()) { throw MLViewError.inputIndexOutsideOfRange }
        case .dataOutput:
            throw MLViewError.dataSetNotRegression
        case .classLabel:     //  Ignores index
            break
        }
        sourceTypeXAxis = source
        sourceIndexXAxis = index
    }
    open func setYAxisSource(_ source: MLViewAxisSource, index: Int) throws {
        switch (source) {
        case .dataInput:
            if (index > classifier.getInputDimension()) { throw MLViewError.inputIndexOutsideOfRange }
        case .dataOutput:
            throw MLViewError.dataSetNotRegression
        case .classLabel:     //  Ignores index
            break
        }
        sourceTypeYAxis = source
        sourceIndexYAxis = index
    }
}


///  Class for showing axis labels
open class MLViewAxisLabel: MLViewItem {
    open var showXAxis : Bool
    open var showYAxis : Bool
    
    //  Drawing offset
    var xOffset : CGFloat = 0.0
    var yOffset : CGFloat = 0.0
    var xAxisHeight : CGFloat = 0.0
    var yAxisWidth : CGFloat = 0.0
    
    //  Axis scaling ranges
    var minX = 0.0
    var maxX = 100.0
    var minY = 0.0
    var maxY = 100.0
    
    //  Axis appearance
    open var XAxisColor = NSColor.gray
    open var YAxisColor = NSColor.gray
    open var MajorTickDivisions = 5
    open var MinorTicksDivisionsPerMajorTick = 5
    open var XAxisMajorTickHeight : CGFloat = 6.0
    open var XAxisMinorTickHeight : CGFloat = 3.0
    open var YAxisMajorTickWidth : CGFloat = 6.0
    open var YAxisMinorTickWidth : CGFloat = 3.0
    open var labelFont = NSFont(name: "Helvetica Neue", size: 10.0)
    open var XAxisLabelDecimalDigits = 2
    open var YAxisLabelDecimalDigits = 2
    
    //  Internal draw variables
    var xLabelHeight : CGFloat = 0.0
    var yMaxLabelWidth : CGFloat = 0.0
    
    public init(showX: Bool, showY: Bool) {
        showXAxis = showX
        showYAxis = showY
    }
    
    open func setOffsets(_ x: CGFloat, y: CGFloat) {
        xOffset = x
        yOffset = y
    }

    open func setColor(_ color: NSColor) {
        XAxisColor = color
        YAxisColor = color
    }
    open func setScale(_ scale: (minX: Double, maxX: Double, minY: Double, maxY: Double)) {
        //  Store the scale
        minX = scale.minX
        maxX = scale.maxX
        minY = scale.minY
        maxY = scale.maxY
    }
    open func draw(_ bounds: CGRect) {
        if (showXAxis) {
            //  Get the label font attributes
            let labelParaStyle = NSMutableParagraphStyle()
            labelParaStyle.lineSpacing = 0.0
            labelParaStyle.alignment = NSTextAlignment.center
            let labelAttributes = [
                NSForegroundColorAttributeName: XAxisColor,
                NSParagraphStyleAttributeName: labelParaStyle,
                NSFontAttributeName: labelFont!
            ] as [String : Any]
            
            let format = String(format: "%%.%df", XAxisLabelDecimalDigits)
            XAxisColor.set()
            let path = NSBezierPath()
            let yPos = yOffset + xLabelHeight + 2.0
            path.move(to: CGPoint(x: bounds.origin.x, y: yPos))
            path.line(to: CGPoint(x: bounds.origin.x + bounds.width, y: yPos))
            path.stroke()
            for i in 0...MajorTickDivisions {
                let xpos = bounds.width * CGFloat(i) / CGFloat(MajorTickDivisions) + bounds.origin.x
                path.removeAllPoints()
                path.move(to: CGPoint(x: xpos, y: yPos))
                path.line(to: CGPoint(x: xpos, y: yPos + XAxisMajorTickHeight))
                path.stroke()
                let value = (maxX - minX) * Double(i) / Double(MajorTickDivisions) + minX
                let label = String(format: format, value)
                let labelSize = label.size(withAttributes: labelAttributes)
                let labelRect = CGRect(x: xpos - labelSize.width * 0.5, y: yPos - 2.0 - labelSize.height, width: labelSize.width, height: labelSize.height)
                label.draw(in: labelRect, withAttributes: labelAttributes)
                if (MinorTicksDivisionsPerMajorTick > 1) {
                    for j in 1..<MinorTicksDivisionsPerMajorTick {
                        let minorXpos = bounds.width * CGFloat(j) / CGFloat(MajorTickDivisions * MinorTicksDivisionsPerMajorTick)  + xpos
                        path.removeAllPoints()
                        path.move(to: CGPoint(x: minorXpos, y: yPos))
                        path.line(to: CGPoint(x: minorXpos, y: yPos + XAxisMinorTickHeight))
                        path.stroke()
                    }
                }
            }
        }
        
        if (showYAxis) {
            //  Get the label font attributes
            let labelParaStyle = NSMutableParagraphStyle()
            labelParaStyle.lineSpacing = 0.0
            labelParaStyle.alignment = NSTextAlignment.right
            let labelAttributes = [
                NSForegroundColorAttributeName: YAxisColor,
                NSParagraphStyleAttributeName: labelParaStyle,
                NSFontAttributeName: labelFont!
            ] as [String : Any]
            
            let format = String(format: "%%.%df", YAxisLabelDecimalDigits)
            YAxisColor.set()
            let path = NSBezierPath()
            let xPos = xOffset + yMaxLabelWidth + 2.0
            path.move(to: CGPoint(x: xPos, y: bounds.origin.y))
            path.line(to: CGPoint(x: xPos, y: bounds.origin.y + bounds.height))
            path.stroke()
            for i in 0...MajorTickDivisions {
                let ypos = bounds.height * CGFloat(i) / CGFloat(MajorTickDivisions) + bounds.origin.y
                path.removeAllPoints()
                path.move(to: CGPoint(x: xPos, y: ypos))
                path.line(to: CGPoint(x: xPos + YAxisMajorTickWidth, y: ypos))
                path.stroke()
                let value = (maxY - minY) * Double(i) / Double(MajorTickDivisions) + minY
                let label = String(format: format, value)
                let labelSize = label.size(withAttributes: labelAttributes)
                let labelRect = CGRect(x: xOffset, y: ypos - (labelSize.height * 0.5), width: yMaxLabelWidth, height: labelSize.height)
                label.draw(in: labelRect, withAttributes: labelAttributes)
                if (MinorTicksDivisionsPerMajorTick > 1) {
                    for j in 1..<MinorTicksDivisionsPerMajorTick {
                        let minorYpos = bounds.height * CGFloat(j) / CGFloat(MajorTickDivisions * MinorTicksDivisionsPerMajorTick)  + ypos
                        path.removeAllPoints()
                        path.move(to: CGPoint(x: xPos, y: minorYpos))
                        path.line(to: CGPoint(x: xPos + YAxisMinorTickWidth, y: minorYpos))
                        path.stroke()
                    }
                }
            }
        }
    }
    open func getScale() -> (minX: Double, maxX: Double, minY: Double, maxY: Double)? {
        return nil
    }
    
    func getXAxisHeight() -> CGFloat {
        //  If not showing one, return 0
        xAxisHeight = 0.0
        if (!showXAxis) { return 0.0 }
        
        //  Get the label font sizing
        let labelParaStyle = NSMutableParagraphStyle()
        labelParaStyle.lineSpacing = 0.0
        labelParaStyle.alignment = NSTextAlignment.center
        let labelAttributes = [
            NSForegroundColorAttributeName: XAxisColor,
            NSParagraphStyleAttributeName: labelParaStyle,
            //NSTextAlignment: textalign,
            NSFontAttributeName: labelFont!
        ] as [String : Any]
        let labelSize = "123.4".size(withAttributes: labelAttributes)
        
        xAxisHeight = XAxisMajorTickHeight + 2.0     //  Tick plus marging
        xAxisHeight += labelSize.height
        xLabelHeight = labelSize.height
        return xAxisHeight
    }
    func getYAxisWidth() -> CGFloat {
        //  If not showing one, return 0
        yAxisWidth = 0.0
        if (!showYAxis) { return 0.0 }
        
        //  Get the label font attributes
        let labelParaStyle = NSMutableParagraphStyle()
        labelParaStyle.lineSpacing = 0.0
        labelParaStyle.alignment = NSTextAlignment.center
        let labelAttributes = [
            NSForegroundColorAttributeName: XAxisColor,
            NSParagraphStyleAttributeName: labelParaStyle,
            //NSTextAlignment: textalign,
            NSFontAttributeName: labelFont!
        ] as [String : Any]
        
        //  Check the width of each label
        yMaxLabelWidth = 0.0
        let format = String(format: "%%.%df", YAxisLabelDecimalDigits)
        for i in 0...MajorTickDivisions {
            let value = (maxY - minY) * Double(i) / Double(MajorTickDivisions) + minY
            let label = String(format: format, value)
            let labelSize = label.size(withAttributes: labelAttributes)
            if (labelSize.width > yMaxLabelWidth) { yMaxLabelWidth = labelSize.width }
        }
        
        yAxisWidth = YAxisMajorTickWidth + 2.0     //  Tick plus marging
        yAxisWidth += yMaxLabelWidth
        return yAxisWidth
    }

    open func setInputVector(_ vector: [Double]) throws {
        //  Not needed for labels
    }
    
    open func setXAxisSource(_ source: MLViewAxisSource, index: Int) throws {
        //  Not needed for labels
    }
    open func setYAxisSource(_ source: MLViewAxisSource, index: Int) throws {
        //  Not needed for labels
    }
}


///  Class for an item in a legend
open class MLLegendItem {
    var label = "item"
    var symbol: MLPlotSymbol?   //  If nil, then line style item
    var lineColor = NSColor.red
    var lineThickness : CGFloat = 1.0
    var itemHeight : CGFloat = 0.0
    
    public init() {
    }
    
    public convenience init(label: String, dataSetPlotItem : MLViewRegressionDataSet) {
        self.init()
        
        self.symbol = dataSetPlotItem.symbol
        self.label = label
    }
    
    public convenience init(label: String, regressionLinePlotItem : MLViewRegressionLine) {
        self.init()
        
        self.symbol = nil
        lineColor = regressionLinePlotItem.lineColor
        lineThickness = regressionLinePlotItem.lineThickness
        
        self.label = label
    }
    
    public convenience init(label: String, plotSymbol : MLPlotSymbol) {
        self.init()
        
        self.symbol = plotSymbol
        self.label = label
    }
    
    open static func createClassLegendArray(_ labelStart: String, classificationDataSet: MLViewClassificationDataSet) -> [MLLegendItem] {
        var array : [MLLegendItem] = []
        
        //  Iterate through each class label
        //!!  currently this is index - need to add label processing everywhere in classification
        var labelIndex = 0
        for plotSymbol in classificationDataSet.symbols {
            let label = labelStart + "\(labelIndex)"
            array.append(MLLegendItem(label: label, plotSymbol : plotSymbol))
            labelIndex += 1
        }
        
        return array
    }
    
}

///  Class for drawing a legend
open class MLViewLegend: MLViewItem {
    
    //  Location
    var location = MLViewLegendLocation.lowerRight
    var xPosition : CGFloat = 0.0
    var yPosition : CGFloat = 0.0
    
    //  Text information
    open var fontColor = NSColor.black
    open var titleFont = NSFont(name: "Helvetica Neue", size: 14.0)
    open var itemFont = NSFont(name: "Helvetica Neue", size: 12.0)
    open var title = ""
    
    //  Items
    var items : [MLLegendItem] = []
    
    public init() {
    }
    
    public convenience init(location: MLViewLegendLocation, title: String) {
        self.init()
        
        self.location = location
        self.title = title
    }
    
    open func addItem(_ item: MLLegendItem) {
        items.append(item)
    }
    
    open func addItems(_ items: [MLLegendItem]) {
        self.items += items
    }
    
    open func setColor(_ color: NSColor) {
        fontColor =  color
    }
    open func setScale(_ scale: (minX: Double, maxX: Double, minY: Double, maxY: Double)) {
        //  Unused
    }
    open func draw(_ bounds: CGRect) {
        //  Get the attributes for drawing
        let paraStyle = NSMutableParagraphStyle()
        paraStyle.lineSpacing = 6.0
        paraStyle.alignment = NSTextAlignment.center
        let titleAttributes = [
            NSForegroundColorAttributeName: fontColor,
            NSParagraphStyleAttributeName: paraStyle,
            //NSTextAlignment: textalign,
            NSFontAttributeName: titleFont!
        ] as [String : Any]
        let labelParaStyle = NSMutableParagraphStyle()
        labelParaStyle.lineSpacing = 6.0
        labelParaStyle.alignment = NSTextAlignment.right
        let labelAttributes = [
            NSForegroundColorAttributeName: fontColor,
            NSParagraphStyleAttributeName: labelParaStyle,
            //NSTextAlignment: textalign,
            NSFontAttributeName: itemFont!
        ] as [String : Any]
        
        //  Get the required size of the legend title
        var titleSize = CGSize.zero
        var legendSize = CGSize.zero
        if (!title.isEmpty) {
            titleSize = title.size(withAttributes: titleAttributes)
            legendSize = titleSize
        }
        
        //  Add all the item sizes
        var maxLabelSize : CGFloat = 0.0
        var maxSymbolSize : CGFloat = 0.0
        for item in items {
            let labelSize = item.label.size(withAttributes: titleAttributes)
            var itemHeight = labelSize.height
            if (labelSize.width > maxLabelSize) { maxLabelSize = labelSize.width }
            if let symbol = item.symbol {
                if (symbol.symbolSize > itemHeight) { itemHeight = symbol.symbolSize }
                if (symbol.symbolSize > maxSymbolSize) { maxSymbolSize = symbol.symbolSize }
            }
            else {
                if (item.lineThickness > itemHeight) { itemHeight = item.lineThickness }
                if (item.lineThickness > maxSymbolSize) { maxSymbolSize = item.lineThickness }
                if (30.0 > maxSymbolSize) { maxSymbolSize = 30.0 }  //  legend line will be at least 30 points long
            }
            item.itemHeight = itemHeight
            legendSize.height += itemHeight + 2.0
        }
        let itemWidth = 2.0 + maxLabelSize + 4.0 + maxSymbolSize + 2.0  //  Include margins
        if (itemWidth > legendSize.width) { legendSize.width = itemWidth }
        
        //  Get the legend position
        switch (location) {
        case .upperLeft:
            xPosition = bounds.origin.x
            yPosition = bounds.height + bounds.origin.y
        case .upperRight:
            xPosition = bounds.width - legendSize.width + bounds.origin.x
            yPosition = bounds.height + bounds.origin.y
        case .lowerLeft:
            xPosition = bounds.origin.x
            yPosition = legendSize.height + bounds.origin.y
        case .lowerRight:
            xPosition = bounds.width - legendSize.width + bounds.origin.x
            yPosition = legendSize.height + bounds.origin.y
        case .custom:
            break   //  position already set
        }
        
        //  If specified, draw the title
        if (!title.isEmpty) {
            let rect = CGRect(x: xPosition, y: yPosition - titleSize.height, width: legendSize.width, height: titleSize.height)
            title.draw(in: rect, withAttributes: titleAttributes)
        }
        
        //  Draw each item
        var labelRect = CGRect(x: xPosition + 2.0, y: yPosition - titleSize.height, width: maxLabelSize, height: 1.0)
        var symbolRect = CGRect(x: xPosition + 2.0 + maxLabelSize + 4.0, y: yPosition - titleSize.height, width: maxSymbolSize, height: 1.0)
        for item in items {
            //  Move the rectangles down
            labelRect.origin.y -= item.itemHeight
            symbolRect.origin.y -= item.itemHeight
            
            //  Set the height of the rectangles to the item height
            labelRect.size.height = item.itemHeight + 2.0
            symbolRect.size.height = item.itemHeight + 2.0
            
            //  Draw the label
            item.label.draw(in: labelRect, withAttributes: labelAttributes)
            
            //  Draw the symbol
            if let symbol = item.symbol {
                let center = CGPoint(x: symbolRect.origin.x + (symbolRect.width * 0.5), y: symbolRect.origin.y + (item.itemHeight * 0.5))
                symbol.drawAt(center)
            }
            else {
                let path = NSBezierPath()
                item.lineColor.setStroke()
                path.lineWidth = item.lineThickness
                let y = symbolRect.origin.y + (item.itemHeight * 0.5)
                path.move(to: CGPoint(x: symbolRect.origin.x + item.lineThickness, y: y))
                path.line(to: CGPoint(x: symbolRect.origin.x + symbolRect.width - item.lineThickness, y: y))
                path.stroke()
            }
        }
    }
    open func getScale() -> (minX: Double, maxX: Double, minY: Double, maxY: Double)? {
        return nil
    }
    
    open func setInputVector(_ vector: [Double]) throws {
        //  Not needed for legends
    }
    
    open func setXAxisSource(_ source: MLViewAxisSource, index: Int) throws {
        //  Not needed for legends
    }
    open func setYAxisSource(_ source: MLViewAxisSource, index: Int) throws {
        //  Not needed for legends
    }
}

///  Class to view Machine Learning data, include data sets, regression lines, classification zones
open class MLView: NSView {
    
    
    //  Machine learning items to plot
    var plotItems : [MLViewItem] = []
    var scalingItem: MLViewItem?
    var margin: CGFloat = 3.0       //  Margin for data and lines, in percent
    
    override public init(frame: CGRect) {
        super.init(frame: frame)
    }
    
    convenience init() {
        self.init(frame: CGRect.zero)
    }
    
    required public init?(coder aDecoder: NSCoder) {
        super.init(coder: aDecoder)
    }
    
    ///  Function to add an item to be plotted
    open func addPlotItem(_ plotItem: MLViewItem) {
        plotItems.append(plotItem)
        setNeedsDisplay(bounds)
    }
    
    ///  Function to set the item that the initial scale will be retrieved from
    open func setInitialScaleItem(_ plotItem: MLViewItem) {
        scalingItem = plotItem
        setNeedsDisplay(bounds)
    }

    override open func draw(_ rect: CGRect) {
        //  draw the background
        NSColor.white.setFill()
        NSRectFill(bounds)
        
        //  Start the initial scaling at 0-100
        var currentScaling = (minX: 0.0, maxX: 100.0, minY: 0.0, maxY: 100.0)
        
        //  If there is an initial item to get the scale from, do so
        if let scalingItem = scalingItem {
            let newScale = scalingItem.getScale()
            if let newScale = newScale {
                currentScaling = newScale
            }
        }
        
        //  See how much room we have to leave for axis labels
        var drawRect = bounds
        let xMargin = (margin * 0.01) * drawRect.size.width
        let yMargin = (margin * 0.01) * drawRect.size.height
        var xAxisLabelSpace : CGFloat = yMargin
        var yAxisLabelSpace : CGFloat = xMargin
        for plotItem in plotItems {
            if plotItem is MLViewAxisLabel {
                let axisLabel = plotItem as! MLViewAxisLabel
                axisLabel.setOffsets(yAxisLabelSpace, y: xAxisLabelSpace)
                xAxisLabelSpace += axisLabel.getXAxisHeight()
                yAxisLabelSpace += axisLabel.getYAxisWidth()
            }
        }
        
        //  Get the draw rectangle without the margins and labels
        drawRect.origin.x += yAxisLabelSpace
        drawRect.origin.y += xAxisLabelSpace
        drawRect.size.width -= xMargin + yAxisLabelSpace
        drawRect.size.height -= yMargin + xAxisLabelSpace
        
        //  Iterate through each plot item
        for plotItem in plotItems {
            //  Set the scaling factors
            plotItem.setScale(currentScaling)
            
            //  Draw the item
            plotItem.draw(drawRect)
            
            //  Get the scaling used for the next item
            let newScale = plotItem.getScale()
            if let newScale = newScale {
                currentScaling = newScale
            }
        }
    }
    
    ///  Method to set the input vector on ALL items
    open func setInputVector(_ vector: [Double]) throws {
        for plotItem in plotItems {
            try plotItem.setInputVector(vector)
        }
    }
    
    ///  Method to set the X axis source on ALL items
    open func setXAxisSource(_ source: MLViewAxisSource, index: Int) throws {
        for plotItem in plotItems {
            try plotItem.setXAxisSource(source, index: index)
        }
    }
    ///  Method to set the Y axis source on ALL items
    open func setYAxisSource(_ source: MLViewAxisSource, index: Int) throws {
        for plotItem in plotItems {
            try plotItem.setYAxisSource(source, index: index)
        }
    }
    
    ///  Routine to round/extend a scale to more 'human-readable' values.
    open static func roundScale(_ min: Double, max: Double, logScale: Bool = false) -> (min: Double, max: Double) {

        var Y : Double
        var Z : Double
        
        //  Set the temporary limit values
        var upper = max
        var lower = min
        
        //  If logarithmic, just round the min and max to the nearest power of 10
        if (logScale) {
            //  Round the upper limit
            if (upper <= 0.0) { upper = 1000.0 }
            Y = log10(upper)
            Z = Double(Int(Y))
            if (Y != Z && Y > 0.0) { Z += 1.0 }
            upper = pow(10.0, Z)
            
            //  round the lower limit
            if (lower <= 0.0) { lower = 0.1}
            Y = log10(lower)
            Z = Double(Int(Y))
            if (Y != Z && Y < 0.0) { Z -= 1.0 }
            lower = pow(10.0, Z)
            
            //  Make sure the limits are not the same
            if (lower == upper) {
                Y = log10(max)
                upper = pow(10.0, Y+1.0)
                lower = pow(10.0, Y-1.0)
            }
            return (min: lower, max: upper)
        }
        
        //  Get the difference between the limits
        var bRoundLimits = true
        var bNegative = false
        while (bRoundLimits) {
            bRoundLimits = false
            let difference = upper - lower
            if (!difference.isFinite) {
                lower = 0.0
                upper = 0.0
                return (min: lower, max: upper)
            }
            
            //  Calculate the upper limit
            if (upper != 0) {
                //  Convert negatives to positives
                bNegative = false
                if (upper < 0.0) {
                    bNegative = true
                    upper *= -1.0
                }
                //  If the limits match, use value for rounding
                if (difference == 0.0) {
                    Z = Double(Int(log10(upper)))
                    if (Z < 0.0) { Z -= 1.0 }
                    Z -= 1.0
                }
                    //  If the limits don't match, use difference for rounding
                else {
                    Z = Double(Int(log10(difference)))
                }
                //  Get the normalized limit
                Y = upper / pow(10.0, Z)
                //  Make sure we don't round down due to value storage limitations
                let NY = Y + DBL_EPSILON * 100.0
                if (Int(log10(Y)) != Int(log10(NY))) {
                    Y = NY * 0.1
                    Z += 1.0
                }
                //  Round by integerizing the normalized number
                if (Y != Double(Int(Y))) {
                    Y = Double(Int(Y))
                    if (!bNegative) {
                        Y += 1.0
                    }
                    upper = Y * pow(10.0, Z)
                }
                if (bNegative) { upper *= -1.0 }
            }
            
            //  Calclate the lower limit
            if (lower != 0) {
                //  Convert negatives to positives
                bNegative = false
                if (lower < 0.0) {
                    bNegative = true
                    lower *= -1.0
                }
                //  If the limits match, use value for rounding
                if (difference == 0.0) {
                    Z = Double(Int(log10(lower)))
                    if (Z < 0.0) { Z -= 1.0 }
                    Z -= 1.0
                }
                    //  If the limits don't match, use difference for rounding
                else {
                    Z = Double(Int(log10(difference)))
                }
                //  Get the normalized limit
                Y = lower / pow(10.0, Z)
                //  Make sure we don't round down due to value storage limitations
                let NY = Y + DBL_EPSILON * 100.0
                if (Int(log10(Y)) != Int(log10(NY))) {
                    Y = NY * 0.1
                    Z += 1.0
                }
                //  Round by integerizing the normalized number
                if (Y != Double(Int(Y))) {
                    Y = Double(Int(Y))
                    if (bNegative) {
                        Y += 1.0
                    }
                    else {
                        if (difference == 0.0) { Y -= 1.0 }
                    }
                    lower = Y * pow(10.0, Z)
                }
                if (bNegative) { lower *= -1.0 }
                
                //  Make sure both are not 0
                if (upper == 0.0 && lower == 0.0) {
                    upper = 1.0;
                    lower = -1.0;
                }
                
                //  If the limits still match offset by a percent each and recalculate
                if (upper == lower) {
                    if (lower > 0.0) {
                        lower *= 0.99
                    }
                    else {
                        lower *= 1.01
                    }
                    if (upper > 0.0) {
                        upper *= 1.01
                    }
                    else {
                        upper *= 0.99
                    }
                    bRoundLimits = true
                }
            }
        }
        
        return (min: lower, max: upper)
    }
}
