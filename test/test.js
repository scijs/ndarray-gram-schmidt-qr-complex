'use strict'

var qr = require('../'),
    ndarray = require('ndarray'),
    pool = require('ndarray-scratch'),
    assert = require('assert'),
    gemm = require('ndarray-blas-gemm-complex'),
    ndt = require('ndarray-tests')


describe("Gram-Schmidt QR", function() {

  it('computes the in-place QR factorization of a square matrix',function() {
    var i,j
    var n=3, m=3
    var A0_r = ndarray([1, 2, 3,4.5, 5, 6, 7, 8, 3], [n,m])
    var A0_i = ndarray([8,-2, 4, -2, 5, 3, 1, 2, 3], [n,m])
    var A_r = ndarray([1, 2, 3,4.5, 5, 6, 7, 8, 3], [n,m])
    var A_i = ndarray([8,-2, 4, -2, 5, 3, 1, 2, 3], [n,m])
    var R_r = pool.zeros( A_r.shape )
    var R_i = pool.zeros( A_r.shape )

    var QR_r = pool.zeros( [m,n] )
    var QR_i = pool.zeros( [m,n] )

    var success = qr(A_r, A_i, R_r, R_i)

    assert( success )

    gemm( A_r, A_i, R_r, R_i, QR_r, QR_i )

    assert( ndt.approximatelyEqual( QR_r, A0_r, 1e-8 ) )
    assert( ndt.approximatelyEqual( QR_i, A0_i, 1e-8 ) )

  })

  it('returns false if the factorization fails',function() {
    var i,j
    var n=2, m=2
    var A_r = ndarray(new Float64Array([1,2,2,4]), [n,m])
    var A_i = ndarray(new Float64Array([0,0,0,0]), [n,m])
    var R_r = pool.zeros( A_r.shape )
    var R_i = pool.zeros( A_r.shape )

    var success = qr(A_r, A_i, R_r, R_i)

    assert( ! success )

  });
});
