'use strict';

var qr = require('../gram-schmidt.js'),
    assert = require('chai').assert,
    ndarray = require('ndarray'),
    pool = require('ndarray-scratch'),
    blas = require('ndarray-blas-level1-complex');


describe("Gram-Schmidt QR", function() {

  it('computes the in-place QR factorization of a square matrix',function() {
    var i,j;
    var n=3, m=3;
    var A0_rdata = [1,  2,  3,4.5,  5,  6,  7,  8,  3];
    var A0_idata = [2, -1,  2, -4,1.5,  0, -7,  2,  1];
    var A_r = ndarray(new Float64Array(A0_rdata), [m,n]);
    var A_i = ndarray(new Float64Array(A0_idata), [m,n]);
    var A0_r = ndarray(new Float64Array(A0_rdata), [m,n]);
    var A0_i = ndarray(new Float64Array(A0_idata), [m,n]);
    var R_r = pool.zeros( A_r.shape, A_r.dtype );
    var R_i = pool.zeros( A_r.shape, A_r.dtype );
    var QR_r = pool.zeros( A_r.shape, A_r.dtype );
    var QR_i = pool.zeros( A_r.shape, A_r.dtype );
    var diff = pool.zeros( A_r.shape, A_r.dtype );

    var success = qr(A_r, A_i, R_r, R_i);

    assert(success);

    // Confirm that all sub-diagonal entries are close to zero:
    for(i=1; i<n; i++) {
      for(j=0; j<i; j++) {
        assert.closeTo( R_r.get(i,j), 0, 1e-8 );
        assert.closeTo( R_i.get(i,j), 0, 1e-8 );
      }
    }

    // Compute the matrix product and confirm it replicates the original:
    for(i=0; i<m; i++) {
      for(j=0; j<n; j++) {
        var d = blas.dotu( A_r.pick(i,null), A_i.pick(i,null), R_r.pick(null,j), R_i.pick(null,j) );
        assert.closeTo( d[0], A0_r.get(i,j), 1e-2 );
        assert.closeTo( d[1], A0_i.get(i,j), 1e-2 );
      }
    }

  });

  it('returns false if the factorization fails',function() {
    var i,j;
    var n=2, m=2;
    var A_r = ndarray(new Float64Array([1,2,2,4]), [m,n]);
    var A_i = ndarray(new Float64Array([1,2,2,4]), [m,n]);
    var R_r = pool.zeros( A_r.shape, A_r.dtype );
    var R_i = pool.zeros( A_r.shape, A_r.dtype );

    var success = qr(A_r, A_i, R_r, R_i);

    assert(success === false);

  });
});
