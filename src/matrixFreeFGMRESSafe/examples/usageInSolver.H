// Example usage inside your Newton solver after rhs_U and rhs_p are available.
// This is not compiled by the library.

#include "MatrixFreeBiCGStab2.H"
#include "ProjectedNewtonOperatorSkeleton.H"

// Unknown correction x = (U_tag, p_tag).  Use fields with homogeneous BCs.
Foam::MatrixFreeBiCGStab2::VpField x("newtonCorrection", U_tag, p_tag);
x.U = U_tag;
x.p = p_tag;

// RHS b = (rhs_U, rhs_p).
Foam::MatrixFreeBiCGStab2::VpField b("newtonRhs", U_tag, p_tag);
b.U = rhs_U;
b.p = rhs_p;

Foam::MatrixFreeBiCGStab2::Controls controls;
controls.tolerance = 1e-8;
controls.relTol = 0.0;
controls.maxIter = 2000;
controls.logEvery = 10;
controls.pressureWeight = 1.0;
controls.name = "newtonBiCGStab2";

ProjectedNewtonOperatorSkeleton A(mesh, U_base, nuEff, p_tag);
Foam::MatrixFreeBiCGStab2 solver(controls);
Foam::MatrixFreeBiCGStab2::Performance perf = solver.solve(A, b, x);

U_tag = x.U;
p_tag = x.p;
