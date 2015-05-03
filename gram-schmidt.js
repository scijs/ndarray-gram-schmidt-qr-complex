'use strict';

var assert = require('assert');

var blas = require('ndarray-blas-level1-complex');

module.exports = function modifiedGramSchmidtQR( A_r, A_i, R_r, R_i ) {

  var i,j, rii, vi_r, vi_i, qi_r, qi_i, vj_r, vj_i, rij;

  assert(A_r.dimension === 2);

  var n = A_r.shape[0];
  //var m = A_r.shape[1];

  for( i=0; i<n; i++ ) {

    // vi = ai
    vi_r = A_r.pick( null, i );
    vi_i = A_i.pick( null, i );

    // rii = ||vi||
    rii = blas.nrm2( vi_r, vi_i );
    if( rii===0 ) { return false; }
    R_r.set(i, i, rii);
    R_i.set(i, i, 0);

    // qi = vi/rii
    qi_r = A_r.pick( null, i );
    qi_i = A_i.pick( null, i );

    blas.cpsc( 1/rii, 0, vi_r, vi_i, qi_r, qi_i );

    for( j=i+1; j<n; j++ ) {
      //rij = qi' * vj
      vj_r = A_r.pick( null, j );
      vj_i = A_i.pick( null, j );

      rij = blas.doth( qi_r, qi_i, vj_r, vj_i );

      R_r.set( i, j, rij[0] );
      R_i.set( i, j, rij[1] );

      // vj = vj - rij * qi
      blas.axpy( -rij[0], -rij[1], qi_r, qi_i, vj_r, vj_i );
    }
  }

  return true;
};
