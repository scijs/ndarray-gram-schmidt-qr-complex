# ndarray-gram-schmidt-qr-complex

[![Build Status](https://travis-ci.org/rreusser/ndarray-gram-schmidt-qr-complex.svg?branch=master)](https://travis-ci.org/rreusser/ndarray-gram-schmidt-qr-complex) [![npm version](https://badge.fury.io/js/ndarray-gram-schmidt-qr-complex.svg)](http://badge.fury.io/js/ndarray-gram-schmidt-qr-complex)

A module for calculating the in-place [QR decomposition of a complex matrix](http://en.wikipedia.org/wiki/QR_decomposition)

## Introduction

The algorithm is the numerically stable variant of the Gram-Schmidt QR decomposition as found on p. 58 of Trefethen and Bau's [Numerical Linear Algebra](http://www.amazon.com/Numerical-Linear-Algebra-Lloyd-Trefethen/dp/0898713617). In pseudocode, the algorithm is:

```
for i = 1 to n
  v_i = a_i

for i = 1 to n
  r_ii = ||v_i||
  q_i = v_i / r_ii

  for j = i+1 to n
    r_ij = q_i' * v_j
    v_j = v_j - r_ij * q_i
```

For real numbers, see [ndarray-gram-schmidt-qr](https://github.com/scijs/ndarray-gram-schmidt-qr).

## Usage

The algorithm currently only calculates the in-place QR decomposition and returns true on successful completion.

```
var qr = require('ndarray-gram-schmidt-qr-complex'),
    pool = require('ndarray-scratch');

var A_r = ndarray( new Float64Array([1,2,7,4,5,1,7,4,9]), [3,3] ),
    A_i = ndarray( new Float64Array([9,3,2,4,4,0,4,1,1]), [3,3] ),
    R_r = pool.zeros( A_r.shape, A_r.dtype );
    R_i = pool.zeros( A_r.shape, A_r.dtype );

qr( A_r, A_i, R_r, R_i );
```

Then the product A * R is approximately equal to the original matrix.

## Credits
(c) 2015 Ricky Reusser. MIT License
