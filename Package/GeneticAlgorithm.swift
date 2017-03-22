//
//  GeneticAlgorithm.swift
//  AIToolbox
//
//  Created by Kevin Coble on 2/26/15.
//  Copyright (c) 2015 Kevin Coble. All rights reserved.
//

import Foundation

#if os(Linux)
    import Glibc
#endif

///  Use this class to do genetic evolution of a population
open class Population
{
    var population = [Genome]()
    
    open var mutationRate = 0.01   //  Default to a 1% mutation rate
    open var sexualReproduction = true    //  Default to sexual reproduction
    
    ///  Create the genome set for your population with this initializer
    public init(populationSize: Int, integerGeneLengths : [Int], doubleGeneLengths : [Int], doubleGeneRanges : [(min : Double, max : Double)])
    {
        for _ in 0..<populationSize {
            let member = Genome(integerGeneLengths: integerGeneLengths, doubleGeneLengths: doubleGeneLengths, doubleGeneRanges: doubleGeneRanges)
            population.append(member)
        }
    }
    
    ///  Use this subscript to get to the population members genetic code
    open subscript(index: Int) -> Genome? {
        if (index < 0 || index > population.count) {return nil}
        return population[index]
    }
    
    
    ///  After running the population through a trial set (setting the scores), create a new population set from the existing one here
    ///  Pass in the expected minimum and maximum scores, so ranking of the popoulation members can be done
    ///  Returns the previous best scoring genome
    open func createNextGeneration(_ expectedMinimumScore: Double, expectedMaximumScore: Double) -> Genome {
        
        //  Scale each of the scores to be in a 1-100 range, and accumulate a total score
        let scale = 99.0 / (expectedMaximumScore - expectedMinimumScore)
        let offset = 1 - (expectedMinimumScore * scale)
        var totalScore = 0.0
        for member in population {
            member.score *= scale
            member.score += offset
            totalScore += member.score
        }
        
        //  Sort the population
        population.sort(by: {$0.score > $1.score})
        
        //  Create the new population with the set parameters
        var newPopulation = [Genome]()
        for _ in population {
            //  Pick the father
#if os(Linux)
            let fatherScore = Double(random()) * totalScore / Double(RAND_MAX)
#else
            let fatherScore = Double(arc4random()) * totalScore / Double(UInt32.max)
#endif
            var fatherIndex = 0
            var totalScoreToIndex = 0.0
            for member in population {
                totalScoreToIndex += member.score
                if (fatherScore < totalScoreToIndex) {break}
                fatherIndex += 1
            }
            
            //  If sexual reproduction, find a mother and mate
            if (sexualReproduction) {
                //  Pick the mother
                var motherIndex = fatherIndex;
                while (motherIndex == fatherIndex) {        //  Make sure it is not the father
                    motherIndex = 0
#if os(Linux)
                    let motherScore = Double(random()) * totalScore / Double(RAND_MAX)
#else
                    let motherScore = Double(arc4random()) * totalScore / Double(UInt32.max)
#endif
                    totalScoreToIndex = 0.0
                    for member in population {
                        totalScoreToIndex += member.score
                        if (motherScore < totalScoreToIndex) {break}
                        motherIndex += 1
                    }
                }
                
                //  Mate
                let newMember = Genome(father: population[fatherIndex], mother: population[motherIndex])
                
                //  Mutate
                newMember.mutateWithProbability(mutationRate)
                
                //  Add
                newPopulation.append(newMember)
            }
            
            //  If asexual, copy the father with mutation
            else {
                let newMember = Genome(copyFrom: population[fatherIndex], mutateWithProbability: mutationRate)
                newPopulation.append(newMember)
            }
        }
        
        //  Set the population to the new one
        let previousBest = population[0]
        population = newPopulation
        
        return previousBest
    }
}

///  Use this class to handle the genetic part of your application's population
open class Genome {
    //  Gene collection
    var integerGeneSet : [IntegerGene]
    var doubleGeneSet : [DoubleGene]
    
    ///  Set the score of this individual here
    open var score = 0.0
    
    public init(integerGeneLengths : [Int], doubleGeneLengths : [Int], doubleGeneRanges : [(min : Double, max : Double)])
    {
        integerGeneSet = []
        for length in integerGeneLengths {
            let gene = IntegerGene(randomOfLength: length)
            integerGeneSet.append(gene)
        }
        doubleGeneSet = []
        for geneIndex in 0..<doubleGeneLengths.count {
            let gene = DoubleGene(randomOfLength: doubleGeneLengths[geneIndex], withRange : doubleGeneRanges[geneIndex])
            doubleGeneSet.append(gene)
        }
    }
    
    ///  Assumes mother and father have genes of the same lengths
    public init (father:Genome, mother:Genome) {
        integerGeneSet = []
        for geneIndex in 0..<father.integerGeneSet.count {
            let gene = father.integerGeneSet[geneIndex].mateWithGene(mother.integerGeneSet[geneIndex])
            integerGeneSet.append(gene)
        }
        doubleGeneSet = []
        for geneIndex in 0..<father.doubleGeneSet.count {
            let gene = father.doubleGeneSet[geneIndex].mateWithGene(mother.doubleGeneSet[geneIndex])
            doubleGeneSet.append(gene)
        }
    }
    
    public init (copyFrom:Genome, mutateWithProbability:Double) {
        integerGeneSet = []
        for geneIndex in 0..<copyFrom.integerGeneSet.count {
            let gene = IntegerGene(copy: copyFrom.integerGeneSet[geneIndex])
            gene.mutateWithProbability(mutateWithProbability)
            integerGeneSet.append(gene)
        }
        doubleGeneSet = []
        for geneIndex in 0..<copyFrom.doubleGeneSet.count {
            let gene = DoubleGene(copy: copyFrom.doubleGeneSet[geneIndex])
            gene.mutateWithProbability(mutateWithProbability)
            doubleGeneSet.append(gene)
        }
    }
    
    open func mutateWithProbability(_ probability: Double) {
        for gene in integerGeneSet {
            gene.mutateWithProbability(probability)
        }
        for gene in doubleGeneSet {
            gene.mutateWithProbability(probability)
        }
    }
    
    ///  Use this member to get the value of an Integer gene allele
    open func integerValueFromGene(_ gene: Int, sequenceIndex: Int) -> UInt32? {
        //  Check the gene number
        if (gene < 0 || gene > integerGeneSet.count) {return nil}
        
        //  Check the allele number
        if (sequenceIndex < 0 || sequenceIndex > integerGeneSet[gene].sequence.count) {return nil}
        
        //  Return the allele
        return integerGeneSet[gene].sequence[sequenceIndex]
    }
    
    ///  Use this member to get the value of an Double gene allele
    open func doubleValueFromGene(_ gene: Int, sequenceIndex: Int) -> Double? {
        //  Check the gene number
        if (gene < 0 || gene > doubleGeneSet.count) {return nil}
        
        //  Check the allele number
        if (sequenceIndex < 0 || sequenceIndex > doubleGeneSet[gene].sequence.count) {return nil}
        
        //  Return the allele
        return doubleGeneSet[gene].sequence[sequenceIndex]
    }
    
    ///  Use this member to initialize an Integer gene to a mutated set of these values
    open func initializeIntegerGene(_ gene: Int, toValues:[UInt32], andMutateWithProbability: Double) -> Bool {
        //  Check the gene number
        if (gene < 0 || gene > integerGeneSet.count) {return false}
        
        //  Check the lengths
        if (toValues.count != integerGeneSet[gene].sequence.count) {return false}
        
        //  Set the gene
        integerGeneSet[gene].sequence = toValues
        
        //  Mutate
        integerGeneSet[gene].mutateWithProbability(andMutateWithProbability)
        
        return true
    }
    
    ///  Use this member to initialize an Integer gene to a mutated set of these values
    open func initializeDoubleGene(_ gene: Int, toValues:[Double], andMutateWithProbability: Double) -> Bool {
        //  Check the gene number
        if (gene < 0 || gene > doubleGeneSet.count) {return false}
        
        //  Check the lengths
        if (toValues.count != doubleGeneSet[gene].sequence.count) {return false}
        
        //  Set the gene
        doubleGeneSet[gene].sequence = toValues
        
        //  Mutate
        doubleGeneSet[gene].mutateWithProbability(andMutateWithProbability)
        
        return true
    }
}


open class IntegerGene
{
    var sequence : [UInt32] = []
    
    init() {
        //  Empty initializer
    }
    
    init(randomOfLength : Int) {
        for _ in 0..<randomOfLength {
#if os(Linux)
            let allele = UInt32(random())
#else
            let allele = arc4random()
#endif
            sequence.append(allele)
        }
    }
    
    init(copy:IntegerGene) {
        for allele in copy.sequence {
            sequence.append(allele)
        }
    }
    
    func mutateWithProbability(_ probability: Double)->Void {
        //  Determine the total number of bits in the gene
        let numBits = sequence.count * 32
        
        //  Get the integer comparison number for the random number generator that matches the mutate probability
#if os(Linux)
        let mutateThreshold = Int(probability * Double(RAND_MAX))
#else
        let mutateThreshold = UInt32(probability * Double(UInt32.max))
#endif
        
        //  Iterate through each bit, mutate if random chance says so
        for bit in 0..<numBits {
#if os(Linux)
            let randNum = random()
#else
            let randNum = arc4random()
#endif
            if (randNum < mutateThreshold) {
                //  Get the allele index and bit mask
                let allele = bit >> 5
                let mask = UInt32(1 << (bit & 0x0000001F))
                
                //  Use exclusive-or to toggle the bit
                sequence[allele] ^= mask
            }
        }
    }
    
    func mateWithGene(_ mate : IntegerGene) -> IntegerGene {
        let newGene = IntegerGene()
        
        //  Get a random length between the two parents gene lengths
        var length = sequence.count
        if (sequence.count > mate.sequence.count) {
            let difference = sequence.count - mate.sequence.count
#if os(Linux)
            length = mate.sequence.count + Int(random() % (difference+1))
#else
            length = mate.sequence.count + Int(arc4random_uniform(UInt32(difference+1)))
#endif
        }
        else if (sequence.count < mate.sequence.count) {
            let difference = mate.sequence.count - sequence.count
#if os(Linux)
            length = sequence.count + Int(random() % (difference+1))
#else
            length = sequence.count + Int(arc4random_uniform(UInt32(difference+1)))
#endif
        }
        
        //  Process each allele
#if os(Linux)
        let compareThreshold = Int(0.5 * Double(RAND_MAX))
#else
        let compareThreshold = UInt32(0.5 * Double(UInt32.max))
#endif
        for i in 0..<length {
            var allele : UInt32
            if (i > sequence.count) {
                allele = mate.sequence[i]
            }
            else if (i > mate.sequence.count) {
                allele = sequence[i]
            }
            else {
#if os(Linux)
                let randNum = random()
#else
                let randNum = arc4random()
#endif
                if (randNum < compareThreshold) {
                    allele = mate.sequence[i]
                }
                else {
                    allele = sequence[i]
                }
            }
            newGene.sequence.append(allele)
        }
        
        return newGene
    }
}



open class DoubleGene
{
    var sequence : [Double] = []
    let range : (min : Double, max : Double)
    
    init(range : (min : Double, max : Double)) {
        self.range = range
        //  Empty initializer
    }
    
    init(randomOfLength : Int, withRange : (min : Double, max : Double)) {
        self.range = withRange
#if os(Linux)
    let multiplier = (range.max - range.min) / Double(RAND_MAX)
#else
    let multiplier = (range.max - range.min) / Double(UInt32.max)
#endif
        for _ in 0..<randomOfLength {
#if os(Linux)
            let allele = Double(random()) * multiplier + range.min
#else
            let allele = Double(arc4random()) * multiplier + range.min
#endif
            sequence.append(allele)
        }
    }
    
    init(copy:DoubleGene) {
        range = copy.range
        for allele in copy.sequence {
            sequence.append(allele)
        }
    }
    
    func mutateWithProbability(_ probability: Double)->Void {
        //  Get the integer comparison number for the random number generator that matches the mutate probability
#if os(Linux)
        let mutateThreshold = Int(probability * Double(RAND_MAX))
#else
        let mutateThreshold = UInt32(probability * Double(UInt32.max))
#endif
        
        //  Iterate through each allele, mutate if random chance says so
        var allele = 0
        while (allele < sequence.count) {
#if os(Linux)
            let randomThreshold = random()
#else
            let randomThreshold = arc4random()
#endif
            if (randomThreshold < mutateThreshold) {
#if os(Linux)
                let randomNum = random()
#else
                let randomNum = arc4random()
#endif
                var modifier = Double(randomNum & 0x0000001F) * (range.max - range.min) / 128.0
                if ((randomNum & 0x00000100) != 0) {modifier *= -1.0}
                sequence[allele] += modifier
                if (sequence[allele] < range.min) {sequence[allele] = range.min}
                if (sequence[allele] > range.max) {sequence[allele] = range.max}
            }
            allele += 1
        }
    }
    
    func mateWithGene(_ mate : DoubleGene) -> DoubleGene {
        let newGene = DoubleGene(range: range)
        
        //  Get a random length between the two parents gene lengths
        var length = sequence.count
        if (sequence.count > mate.sequence.count) {
            let difference = sequence.count - mate.sequence.count
#if os(Linux)
            length = mate.sequence.count + Int(random() % (difference+1))
#else
            length = mate.sequence.count + Int(arc4random_uniform(UInt32(difference+1)))
#endif
        }
        else if (sequence.count < mate.sequence.count) {
            let difference = mate.sequence.count - sequence.count
#if os(Linux)
            length = sequence.count + Int(random() % (difference+1))
#else
            length = sequence.count + Int(arc4random_uniform(UInt32(difference+1)))
#endif
        }
        
        //  Process each allele
#if os(Linux)
        let compareThreshold = Int(0.5 * Double(RAND_MAX))
#else
        let compareThreshold = UInt32(0.5 * Double(UInt32.max))
#endif
        for i in 0..<length {
            var allele : Double
            if (i > sequence.count) {
                allele = mate.sequence[i]
            }
            else if (i > mate.sequence.count) {
                allele = sequence[i]
            }
            else {
#if os(Linux)
                let randNum = random()
#else
                let randNum = arc4random()
#endif
                if (randNum < compareThreshold) {
                    allele = mate.sequence[i]
                }
                else {
                    allele = sequence[i]
                }
            }
            newGene.sequence.append(allele)
        }
        
        return newGene
    }
}
