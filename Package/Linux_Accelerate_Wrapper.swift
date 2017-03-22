//
//  Linux_Accelerate_Wrapper.swift
//  AIToolbox
//
//  Created by Kevin Coble on 3/20/17.
//  Copyright © 2017 Kevin Coble. All rights reserved.
//
//  This file emulates the Accelerate routines used by AIToolbox.
//  It would be better if vector routines were used in place

#if os(Linux)
typealias vDSP_Length = UInt
typealias vDSP_Stride = Int

//  Double-precision real vector-scalar multiply
func vDSP_vsmulD(_ __A: UnsafePointer<Double>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Double>, _ __C: UnsafeMutablePointer<Double>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    var A_Offset = 0
    var C_Offset = 0
    for _ in 0..<__N {
        __C[C_Offset] = __A[A_Offset] * __B.pointee
        A_Offset += __IA
        C_Offset += __IC
    }
}

    
//  Vector multiply and add; double precision.
func vDSP_vmaD(_ __A: UnsafePointer<Double>,
               _ __IA: vDSP_Stride,
               _ __B: UnsafePointer<Double>,
               _ __IB: vDSP_Stride,
               _ __C: UnsafePointer<Double>,
               _ __IC: vDSP_Stride,
               _ __D: UnsafeMutablePointer<Double>,
               _ __ID: vDSP_Stride,
               _ __N: vDSP_Length){
    
    var A_Offset = 0
    var B_Offset = 0
    var C_Offset = 0
    var D_Offset = 0
    for _ in 0..<__N {
        __D[D_Offset] = __A[A_Offset] * __B[B_Offset] + __C[C_Offset]
        A_Offset += __IA
        B_Offset += __IB
        C_Offset += __IC
        D_Offset += __ID
    }
}

//  Double-precision real vector-scalar multiply and vector add
func vDSP_vsmaD(_ __A: UnsafePointer<Double>,
                _ __IA: vDSP_Stride,
                _ __B: UnsafePointer<Double>,
                _ __C: UnsafePointer<Double>,
                _ __IC: vDSP_Stride,
                _ __D: UnsafeMutablePointer<Double>,
                _ __ID: vDSP_Stride,
                _ __N: vDSP_Length){
    
    var A_Offset = 0
    var C_Offset = 0
    var D_Offset = 0
    for _ in 0..<__N {
        __D[D_Offset] = __A[A_Offset] * __B.pointee + __C[C_Offset]
        A_Offset += __IA
        C_Offset += __IC
        D_Offset += __ID
    }
}

//  Adds two vectors; double precision
func vDSP_vaddD(_ __A: UnsafePointer<Double>,
                _ __IA: vDSP_Stride,
                _ __B: UnsafePointer<Double>,
                _ __IB: vDSP_Stride,
                _ __C: UnsafeMutablePointer<Double>,
                _ __IC: vDSP_Stride,
                _ __N: vDSP_Length) {
    
    var A_Offset = 0
    var B_Offset = 0
    var C_Offset = 0
    for _ in 0..<__N {
        __C[C_Offset] = __A[A_Offset] + __B[B_Offset]
        A_Offset += __IA
        B_Offset += __IB
        C_Offset += __IC
    }
}

//  Vector subtract; double precision
func vDSP_vsubD(_ __B: UnsafePointer<Double>,
                _ __IB: vDSP_Stride,
                _ __A: UnsafePointer<Double>,
                _ __IA: vDSP_Stride,
                _ __C: UnsafeMutablePointer<Double>,
                _ __IC: vDSP_Stride,
                _ __N: vDSP_Length) {
    
    var A_Offset = 0
    var B_Offset = 0
    var C_Offset = 0
    for _ in 0..<__N {
        __C[C_Offset] = __A[A_Offset] - __B[B_Offset]
        A_Offset += __IA
        B_Offset += __IB
        C_Offset += __IC
    }
}

//  Vector multiplication; double precision
func vDSP_vmulD(_ __A: UnsafePointer<Double>,
                _ __IA: vDSP_Stride,
                _ __B: UnsafePointer<Double>,
                _ __IB: vDSP_Stride,
                _ __C: UnsafeMutablePointer<Double>,
                _ __IC: vDSP_Stride,
                _ __N: vDSP_Length) {
    var A_Offset = 0
    var B_Offset = 0
    var C_Offset = 0
    for _ in 0..<__N {
        __C[C_Offset] = __A[A_Offset] * __B[B_Offset]
        A_Offset += __IA
        B_Offset += __IB
        C_Offset += __IC
    }
}
 
//  Divide scalar by vector; double precision.
func vDSP_svdivD(_ __A: UnsafePointer<Double>,
                 _ __B: UnsafePointer<Double>,
                 _ __IB: vDSP_Stride,
                 _ __C: UnsafeMutablePointer<Double>,
                 _ __IC: vDSP_Stride,
                 _ __N: vDSP_Length) {
    
    var B_Offset = 0
    var C_Offset = 0
    for _ in 0..<__N {
        __C[C_Offset] = __A.pointee / __B[B_Offset]
        B_Offset += __IB
        C_Offset += __IC
    }
}

//  Vector-scalar add; double precision.
func vDSP_vsaddD(_ __A: UnsafePointer<Double>,
                     _ __IA: vDSP_Stride,
                     _ __B: UnsafePointer<Double>,
                     _ __C: UnsafeMutablePointer<Double>,
                     _ __IC: vDSP_Stride,
                     _ __N: vDSP_Length) {
    
    var A_Offset = 0
    var C_Offset = 0
    for _ in 0..<__N {
        __C[C_Offset] = __A[A_Offset] + __B.pointee
        A_Offset += __IA
        C_Offset += __IC
    }
}
    
//  Computes the squared values of vector A and leaves the result in vector C; double precision.
func vDSP_vsqD(_ __A: UnsafePointer<Double>,
               _ __IA: vDSP_Stride,
               _ __C: UnsafeMutablePointer<Double>,
               _ __IC: vDSP_Stride,
               _ __N: vDSP_Length) {
    
    var A_Offset = 0
    var C_Offset = 0
    for _ in 0..<__N {
        __C[C_Offset] = __A[A_Offset] * __A[A_Offset]
        A_Offset += __IA
        C_Offset += __IC
    }
}

//  Computes the dot or scalar product of vectors A and B and leaves the result in scalar *C; double precision
func vDSP_dotprD(_ __A: UnsafePointer<Double>,
                 _ __IA: vDSP_Stride,
                 _ __B: UnsafePointer<Double>,
                 _ __IB: vDSP_Stride,
                 _ __C: UnsafeMutablePointer<Double>,
                 _ __N: vDSP_Length) {
    
    var A_Offset = 0
    var B_Offset = 0
    var sum = 0.0
    for _ in 0..<__N {
        sum += __A[A_Offset] * __B[B_Offset]
        A_Offset += __IA
        B_Offset += __IB
    }
    __C.pointee = sum
}

    
//  Square of the Euclidean distance between two points in N-dimensional space, each defined by a real vector; double precision
func vDSP_distancesqD(_ __A: UnsafePointer<Double>,
                      _ __IA: vDSP_Stride,
                      _ __B: UnsafePointer<Double>,
                      _ __IB: vDSP_Stride,
                      _ __C: UnsafeMutablePointer<Double>,
                      _ __N: vDSP_Length) {
    
    var A_Offset = 0
    var B_Offset = 0
    var sum = 0.0
    for _ in 0..<__N {
        let diff = __A[A_Offset] - __B[B_Offset]
        sum += diff * diff
        A_Offset += __IA
        B_Offset += __IB
    }
    __C.pointee = sum
}

//  Performs an out-of-place multiplication of two matrices; double precision
func vDSP_mmulD(_ __A: UnsafePointer<Double>,
                _ __IA: vDSP_Stride,
                _ __B: UnsafePointer<Double>,
                _ __IB: vDSP_Stride,
                _ __C: UnsafeMutablePointer<Double>,
                _ __IC: vDSP_Stride,
                _ __M: vDSP_Length,
                _ __N: vDSP_Length,
                _ __P: vDSP_Length) {
    
    var A_Offset = 0
    for m in 0..<__M {
        var C_Offset = Int(m * __N) * __IC
        for n in 0..<__N {
            __C[C_Offset] = 0.0
            A_Offset = Int(m * __P) * __IA
            for p in 0..<__P {
                __C[C_Offset] += __A[A_Offset] * __B[Int((p * __N) + n) * __IB]
                A_Offset += __IA
            }
            C_Offset += __IC
        }
    }
}
#endif


