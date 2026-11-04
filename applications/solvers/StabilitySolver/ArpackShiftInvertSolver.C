#include "ArpackShiftInvertSolver.H"
#include "Pstream.H"
#include "fvcFlux.H"
#include "fvcDiv.H"
#include "fvcGrad.H"
#include "fvmLaplacian.H"
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace Foam
{
namespace matrixFreeStability
{

namespace
{
inline bool finiteComplex(const std::complex<double>& z)
{
    return std::isfinite(z.real()) && std::isfinite(z.imag());
}

void complexScaleFrom
(
    ComplexFlow& dst,
    const std::complex<double>& alpha,
    const ComplexFlow& src
)
{
    const scalar ar = scalar(alpha.real());
    const scalar ai = scalar(alpha.imag());

    copyFlow(dst.real, src.real);
    scaleFlow(dst.real, ar);
    axpyFlow(dst.real, -ai, src.imag);

    copyFlow(dst.imag, src.real);
    scaleFlow(dst.imag, ai);
    axpyFlow(dst.imag, ar, src.imag);
}

void complexAxpy
(
    ComplexFlow& y,
    const std::complex<double>& alpha,
    const ComplexFlow& x
)
{
    const scalar ar = scalar(alpha.real());
    const scalar ai = scalar(alpha.imag());

    axpyFlow(y.real, ar, x.real);
    axpyFlow(y.real, -ai, x.imag);

    axpyFlow(y.imag, ai, x.real);
    axpyFlow(y.imag, ar, x.imag);
}

scalar globalSum(const scalar local)
{
    return returnReduce(local, sumOp<scalar>());
}

// Local part of max|div(phi)| for a conservative face flux.
scalar localMaxAbsDivPhi(const surfaceScalarField& phi)
{
    tmp<volScalarField> tDiv = fvc::div(phi);
    const scalarField& d = tDiv().primitiveField();

    scalar localMax = 0.0;
    forAll(d, celli)
    {
        localMax = max(localMax, mag(d[celli]));
    }

    return localMax;
}

// Local part of the same diagnostic if the flux is reconstructed from cell U.
// This is intentionally distinct from localMaxAbsDivPhi(q.phi): after the
// shifted saddle-point solve, q.phi is the pressure-corrected conservative
// flux and need not equal fvc::flux(q.U).
scalar localMaxAbsDivFluxOfU(const volVectorField& U)
{
    tmp<surfaceScalarField> tPhi = fvc::flux(U);
    return localMaxAbsDivPhi(tPhi());
}

// Local part of the check that the perturbation satisfies homogeneous velocity
// boundary conditions on the physical walls.
scalar localMaxBoundaryMagU(const volVectorField& U)
{
    scalar localMax = 0.0;

    forAll(U.boundaryField(), patchi)
    {
        // Processor/cyclic patches are internal coupling interfaces, not
        // physical cavity walls.  Gelfgat's homogeneous perturbation BC
        // applies to the physical walls only.
        if (U.boundaryField()[patchi].coupled()) continue;

        const vectorField& Ub = U.boundaryField()[patchi];
        forAll(Ub, facei)
        {
            localMax = max(localMax, mag(Ub[facei]));
        }
    }

    return localMax;
}

// All constraint diagnostics of one Arnoldi step in a single collective.
void globalMaxInPlace(scalar* values, const label n)
{
    static_assert
    (
        sizeof(scalar) == sizeof(double),
        "Batched diagnostics require double precision"
    );

    if (n <= 0 || !Pstream::parRun()) return;

    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wold-style-cast"
    MPI_Allreduce
    (
        MPI_IN_PLACE,
        values,
        int(n),
        MPI_DOUBLE,
        MPI_MAX,
        MPI_COMM_WORLD
    );
    #pragma GCC diagnostic pop
}

// Normalize the exact distributed complex vector that will be supplied to
// PARPACK as RESID.  Packing already includes sqrt(V/Vmean), so this is the
// same Euclidean normalization used by PARPACK and corresponds to the chosen
// FV velocity metric.  A single global scalar scaling preserves solenoidality.
scalar normalizePackedVelocityVector
(
    std::complex<double>* x,
    const label nLocal
)
{
    scalar localNorm2 = 0.0;
    for (label i=0; i<nLocal; ++i)
    {
        localNorm2 += scalar(std::norm(x[i]));
    }

    const scalar norm = Foam::sqrt(max(globalSum(localNorm2), scalar(0)));
    if (norm <= VSMALL)
    {
        FatalErrorInFunction
            << "Cannot normalize a zero PARPACK starting vector"
            << exit(FatalError);
    }

    const double invNorm = 1.0/double(norm);
    for (label i=0; i<nLocal; ++i)
    {
        x[i] *= invNorm;
    }

    return norm;
}

}

ArpackShiftInvertSolver::ArpackShiftInvertSolver
(
    const fvMesh& mesh,
    ShiftedMALSolver& shiftInvertSolver,
    const volVectorField& Utemplate,
    const surfaceScalarField& phitemplate,
    const volScalarField& ptemplate,
    const ArpackArnoldiControls& controls
)
:
    mesh_(mesh),
    shiftInvertSolver_(shiftInvertSolver),
    controls_(controls),
    volumeWeightScale_(1.0),
    cellWeight_(mesh.nCells(), 1.0),
    invCellWeight_(mesh.nCells(), 1.0),
    arpackInput_("arpackInput", Utemplate, phitemplate, ptemplate),
    arpackOutput_("arpackOutput", Utemplate, phitemplate, ptemplate),
    velocityMode_("arnoldiVelocityMode", Utemplate, phitemplate, ptemplate),
    imageMode_("arnoldiImageMode", Utemplate, phitemplate, ptemplate),
    scaledWork_("arnoldiScaledWork", Utemplate, phitemplate, ptemplate),
    jacobianWork_("arnoldiJacobianWork", Utemplate, phitemplate, ptemplate),
    massWork_("arnoldiMassWork", Utemplate, phitemplate, ptemplate),
    residualWork_("arnoldiResidualWork", Utemplate, phitemplate, ptemplate)
{
    // PARPACK Euclidean metric must match the normalized FV metric
    //
    //     w_i = V_i/Vmean,
    //     1/Vmean = Ncells_global/sum(V_i).
    //
    // Therefore packing uses sqrt(V_i/Vmean).
    const scalarField& V = mesh_.V();

    scalar localVolume = 0.0;
    forAll(V, celli)
    {
        localVolume += V[celli];
    }

    const scalar globalVolume =
        returnReduce(localVolume, sumOp<scalar>());

    const label globalNCells =
        returnReduce(mesh_.nCells(), sumOp<label>());

    if (globalNCells <= 0 || globalVolume <= VSMALL)
    {
        FatalErrorInFunction
            << "Cannot construct normalized PARPACK volume metric"
            << exit(FatalError);
    }

    volumeWeightScale_ =
        scalar(globalNCells)/globalVolume;

    if (controls_.volumeWeighting)
    {
        forAll(cellWeight_, celli)
        {
            cellWeight_[celli] =
                Foam::sqrt(max(volumeWeightScale_*V[celli], VSMALL));
            invCellWeight_[celli] = 1.0/cellWeight_[celli];
        }
    }

    if (Pstream::master())
    {
        Info<< "PARPACK velocity metric: normalized cell-volume weighting "
            << "V/Vmean" << nl;
    }

    zeroComplexFlow(arpackInput_);
    zeroComplexFlow(arpackOutput_);
    zeroComplexFlow(velocityMode_);
    zeroComplexFlow(imageMode_);
    zeroComplexFlow(scaledWork_);
    zeroComplexFlow(jacobianWork_);
    zeroComplexFlow(massWork_);
    zeroComplexFlow(residualWork_);
}

label ArpackShiftInvertSolver::localSize() const
{
    // PARPACK state is velocity only: three Cartesian components/cell.
    return 3*mesh_.nCells();
}

label ArpackShiftInvertSolver::globalSize() const
{
    return returnReduce(localSize(), sumOp<label>());
}

arpack::which ArpackShiftInvertSolver::whichOption() const
{
    if (controls_.which == "LM") return arpack::which::largest_magnitude;
    if (controls_.which == "SM") return arpack::which::smallest_magnitude;
    if (controls_.which == "LR") return arpack::which::largest_real;
    if (controls_.which == "SR") return arpack::which::smallest_real;
    if (controls_.which == "LI") return arpack::which::largest_imaginary;
    if (controls_.which == "SI") return arpack::which::smallest_imaginary;

    FatalErrorInFunction
        << "Unsupported arnoldiWhich=" << controls_.which << nl
        << "Use LM, SM, LR, SR, LI or SI. For the already transformed "
        << "shift-invert operator, LM is the correct default."
        << exit(FatalError);

    return arpack::which::largest_magnitude;
}

void ArpackShiftInvertSolver::unpackVelocity
(
    const std::complex<double>* x,
    ComplexFlow& q
) const
{
    zeroComplexFlow(q);

    vectorField& Ur = q.real.U.primitiveFieldRef();
    vectorField& Ui = q.imag.U.primitiveFieldRef();

    forAll(Ur, celli)
    {
        const scalar invWeight = invCellWeight_[celli];
        for (direction cmpt=0; cmpt<vector::nComponents; ++cmpt)
        {
            const label i = 3*celli + label(cmpt);
            Ur[celli][cmpt] = scalar(x[i].real())*invWeight;
            Ui[celli][cmpt] = scalar(x[i].imag())*invWeight;
        }
    }

    setHomogeneousVelocityBoundary(q);
    replacePhiWithFluxOfU(q);
    zeroVolScalar(q.real.p);
    zeroVolScalar(q.imag.p);
}

void ArpackShiftInvertSolver::packVelocity
(
    const ComplexFlow& q,
    std::complex<double>* x
) const
{
    const vectorField& Ur = q.real.U.primitiveField();
    const vectorField& Ui = q.imag.U.primitiveField();

    forAll(Ur, celli)
    {
        const scalar weight = cellWeight_[celli];
        for (direction cmpt=0; cmpt<vector::nComponents; ++cmpt)
        {
            const label i = 3*celli + label(cmpt);
            x[i] = std::complex<double>
            (
                double(weight*Ur[celli][cmpt]),
                double(weight*Ui[celli][cmpt])
            );
        }
    }
}

void ArpackShiftInvertSolver::fillStartingVector(std::complex<double>* x)
{
    const vectorField& C = mesh_.C();

    forAll(C, celli)
    {
        const vector& c = C[celli];
        const scalar basePhase =
            0.731*c.x() + 1.213*c.y() + 0.517*c.z() + 0.137;
        const scalar weight = cellWeight_[celli];

        for (direction cmpt=0; cmpt<vector::nComponents; ++cmpt)
        {
            const scalar phase = basePhase + 0.619*scalar(cmpt + 1);
            const scalar re = std::sin(phase) + 0.23*std::cos(1.7*phase);
            const scalar im = std::cos(1.31*phase) - 0.19*std::sin(0.73*phase);
            x[3*celli + label(cmpt)] = std::complex<double>
            (
                double(weight*re),
                double(weight*im)
            );
        }
    }
}

void ArpackShiftInvertSolver::projectStartingVector
(
    std::complex<double>* x
)
{
    // PARPACK stores velocity only.  The Arnoldi basis must start inside the
    // same discrete homogeneous solenoidal velocity space into which the
    // transformed operator maps:
    //
    //     T_sigma = P_U (J-sigma B)^(-1) E_U .
    //
    // A generic deterministic velocity seed is not discretely solenoidal.
    // If that non-solenoidal vector is used as v1, Arnoldi orthogonalization
    // can reintroduce its divergence into every later basis vector even though
    // each individual shift-invert output satisfies continuity.  Therefore
    // apply the ACTUAL transformed operator once before PARPACK starts and use
    // its velocity output as the initial residual/vector.
    //
    // This is a one-time startup solve only.  It is not counted as a PARPACK
    // OP application and it does not change the operator used by PARPACK.

    unpackVelocity(x, arpackInput_);

    scalar before[4] =
    {
        localMaxAbsDivFluxOfU(arpackInput_.real.U),
        localMaxAbsDivFluxOfU(arpackInput_.imag.U),
        localMaxBoundaryMagU(arpackInput_.real.U),
        localMaxBoundaryMagU(arpackInput_.imag.U)
    };
    globalMaxInPlace(before, 4);

    const scalar divBeforeReal = before[0];
    const scalar divBeforeImag = before[1];
    const scalar wallBeforeReal = before[2];
    const scalar wallBeforeImag = before[3];

    if (Pstream::master())
    {
        Info<< "Building solenoidal PARPACK starting vector with one "
            << "shift-invert application" << nl
            << "  raw max|div(flux(U))|=("
            << divBeforeReal << ',' << divBeforeImag << ')' << nl
            << "  raw max|U_physicalBoundary|=("
            << wallBeforeReal << ',' << wallBeforeImag << ')' << nl;
    }

    label innerIterations = 0;
    const bool converged = shiftInvertSolver_.applyShiftInvert
    (
        arpackInput_,
        arpackOutput_,
        innerIterations,
        controls_.innerPrintInterval,
        true
    );

    if (!converged || !shiftInvertSolver_.lastSolveConverged())
    {
        FatalErrorInFunction
            << "Startup shift-invert application used to build the "
            << "solenoidal PARPACK starting vector did not converge." << nl
            << "inner iterations = " << innerIterations << nl
            << "initial residual = "
            << shiftInvertSolver_.lastInitialResidual() << nl
            << "final true residual = "
            << shiftInvertSolver_.lastTrueResidual() << nl
            << "required target = " << shiftInvertSolver_.lastTarget() << nl
            << "sigma = " << shiftInvertSolver_.sigmaReal()
            << " + i(" << shiftInvertSolver_.sigmaImag() << ")"
            << exit(FatalError);
    }

    scalar after[6] =
    {
        localMaxAbsDivPhi(arpackOutput_.real.phi),
        localMaxAbsDivPhi(arpackOutput_.imag.phi),
        localMaxAbsDivFluxOfU(arpackOutput_.real.U),
        localMaxAbsDivFluxOfU(arpackOutput_.imag.U),
        localMaxBoundaryMagU(arpackOutput_.real.U),
        localMaxBoundaryMagU(arpackOutput_.imag.U)
    };
    globalMaxInPlace(after, 6);

    const scalar divAfterPhiReal = after[0];
    const scalar divAfterPhiImag = after[1];
    const scalar divAfterFluxUReal = after[2];
    const scalar divAfterFluxUImag = after[3];
    const scalar wallAfterReal = after[4];
    const scalar wallAfterImag = after[5];

    if (Pstream::master())
    {
        Info<< "  startup shift-invert iterations=" << innerIterations << nl
            << "  startup finalTrueResidual="
            << shiftInvertSolver_.lastTrueResidual()
            << " target=" << shiftInvertSolver_.lastTarget() << nl
            << "  shift-invert output max|div(phi)|=("
            << divAfterPhiReal << ',' << divAfterPhiImag << ')' << nl
            << "  shift-invert output max|div(flux(U))|=("
            << divAfterFluxUReal << ',' << divAfterFluxUImag
            << ") [diagnostic only]" << nl
            << "  shift-invert output max|U_physicalBoundary|=("
            << wallAfterReal << ',' << wallAfterImag << ')' << nl;
    }

    // Pack the solenoidal physical velocity returned by the saddle-point
    // solve, then normalize the exact distributed vector supplied to PARPACK.
    // A single scalar normalization preserves the solenoidal subspace.
    packVelocity(arpackOutput_, x);
    const scalar preNormalizationNorm =
        normalizePackedVelocityVector(x, localSize());

    // HARD ROUND-TRIP CHECK of the exact normalized representation PARPACK
    // receives: physical startup output -> packed RESID -> normalized RESID ->
    // unpacked U.  Because unpackVelocity reconstructs phi=fvc::flux(U), the
    // final divergence below is the one relevant to the velocity-only Arnoldi
    // basis representation.
    unpackVelocity(x, scaledWork_);

    scalar roundTrip[4] =
    {
        localMaxAbsDivFluxOfU(scaledWork_.real.U),
        localMaxAbsDivFluxOfU(scaledWork_.imag.U),
        localMaxBoundaryMagU(scaledWork_.real.U),
        localMaxBoundaryMagU(scaledWork_.imag.U)
    };
    globalMaxInPlace(roundTrip, 4);

    const scalar finalDivReal = roundTrip[0];
    const scalar finalDivImag = roundTrip[1];
    const scalar finalWallReal = roundTrip[2];
    const scalar finalWallImag = roundTrip[3];

    const bool startPass =
        finalDivReal <= controls_.constraintTolerance
     && finalDivImag <= controls_.constraintTolerance
     && finalWallReal <= controls_.constraintTolerance
     && finalWallImag <= controls_.constraintTolerance;

    if (Pstream::master())
    {
        Info<< "FINAL PARPACK starting-vector round-trip check:" << nl
            << "  packed norm before explicit normalization = "
            << preNormalizationNorm << nl
            << "  normalized packed norm                    = 1" << nl
            << "  max|div(flux(U))|=("
            << finalDivReal << ',' << finalDivImag << ')' << nl
            << "  max|U_physicalBoundary|=("
            << finalWallReal << ',' << finalWallImag << ')' << nl
            << "  constraint tolerance=" << controls_.constraintTolerance << nl
            << "  STARTING VECTOR STATUS: "
            << (startPass ? "PASS" : "FAIL") << nl;
    }

    if (!startPass && controls_.abortOnConstraintViolation)
    {
        FatalErrorInFunction
            << "The exact normalized PARPACK starting vector is not inside "
            << "the required discrete homogeneous solenoidal space." << nl
            << "max|div(flux(U_real))| = " << finalDivReal << nl
            << "max|div(flux(U_imag))| = " << finalDivImag << nl
            << "max|U_real on physical walls| = " << finalWallReal << nl
            << "max|U_imag on physical walls| = " << finalWallImag << nl
            << "tolerance = " << controls_.constraintTolerance << nl
            << exit(FatalError);
    }
}

void ArpackShiftInvertSolver::applyArnoldiOperator
(
    const std::complex<double>* x,
    std::complex<double>* y,
    label& operatorApplications,
    label& totalInnerIterations,
    label& maximumInnerIterations
)
{
    unpackVelocity(x, arpackInput_);

    const label nextOperatorApplication = operatorApplications + 1;
    const bool printThisApplication =
        controls_.printInterval > 0
     &&
        (
            nextOperatorApplication == 1
         || nextOperatorApplication % controls_.printInterval == 0
        );

    // Diagnostics for the incoming PARPACK velocity basis vector.
    // unpackVelocity() has already imposed homogeneous velocity BCs and set
    // input phi=fvc::flux(U).  Because v1 is explicitly solenoidalized before
    // pznaupd starts and every OP output lies in the same discrete constraint
    // space, Arnoldi orthogonalization/restarts should preserve that space up
    // to numerical/inexact-solve error.  Therefore both input divergence and
    // physical-wall velocity are hard constraints when the guard is enabled.
    scalar inputDivFluxUReal = 0.0;
    scalar inputDivFluxUImag = 0.0;
    scalar inputBoundaryUReal = 0.0;
    scalar inputBoundaryUImag = 0.0;

    // Always evaluate the constraint when the hard guard is enabled.  Printing
    // may be less frequent, but verification is then performed on every basis
    // vector supplied by PARPACK.
    if (printThisApplication || controls_.abortOnConstraintViolation)
    {
        scalar in[4] =
        {
            localMaxAbsDivPhi(arpackInput_.real.phi),
            localMaxAbsDivPhi(arpackInput_.imag.phi),
            localMaxBoundaryMagU(arpackInput_.real.U),
            localMaxBoundaryMagU(arpackInput_.imag.U)
        };
        globalMaxInPlace(in, 4);

        inputDivFluxUReal = in[0];
        inputDivFluxUImag = in[1];
        inputBoundaryUReal = in[2];
        inputBoundaryUImag = in[3];
    }

    const bool inputConstraintPass =
        inputDivFluxUReal <= controls_.constraintTolerance
     && inputDivFluxUImag <= controls_.constraintTolerance
     && inputBoundaryUReal <= controls_.constraintTolerance
     && inputBoundaryUImag <= controls_.constraintTolerance;

    if (!inputConstraintPass && controls_.abortOnConstraintViolation)
    {
        FatalErrorInFunction
            << "PARPACK Arnoldi basis vector entering OP application "
            << nextOperatorApplication
            << " left the required discrete homogeneous solenoidal space." << nl
            << "max|div(flux(U_real))| = " << inputDivFluxUReal << nl
            << "max|div(flux(U_imag))| = " << inputDivFluxUImag << nl
            << "max|U_real on physical walls| = " << inputBoundaryUReal << nl
            << "max|U_imag on physical walls| = " << inputBoundaryUImag << nl
            << "tolerance = " << controls_.constraintTolerance << nl
            << exit(FatalError);
    }

    label innerIterations = 0;

    // Keep the inner shifted-FGMRES progress visible.  A PARPACK OP can take
    // hundreds of inner iterations; suppressing this output makes a healthy
    // run look frozen.
    const bool converged = shiftInvertSolver_.applyShiftInvert
    (
        arpackInput_,
        arpackOutput_,
        innerIterations,
        controls_.innerPrintInterval,
        true
    );

    if (!converged || !shiftInvertSolver_.lastSolveConverged())
    {
        FatalErrorInFunction
            << "Shift-invert OP application failed to converge. PARPACK must "
            << "not be given an inaccurate (J-sigma B)^(-1)B product." << nl
            << "inner iterations = " << innerIterations << nl
            << "initial residual = " << shiftInvertSolver_.lastInitialResidual() << nl
            << "final true residual = " << shiftInvertSolver_.lastTrueResidual() << nl
            << "required target = " << shiftInvertSolver_.lastTarget() << nl
            << "sigma = " << shiftInvertSolver_.sigmaReal()
            << " + i(" << shiftInvertSolver_.sigmaImag() << ")"
            << exit(FatalError);
    }

    // The shifted saddle-point solve should enforce div(phi')=0 in the output.
    // We also report div(fvc::flux(U')) separately.  These two quantities are
    // not required to be identical in the collocated formulation because phi'
    // is the pressure-corrected conservative face flux.
    scalar outputDivPhiReal = 0.0;
    scalar outputDivPhiImag = 0.0;
    scalar outputDivFluxUReal = 0.0;
    scalar outputDivFluxUImag = 0.0;
    scalar outputBoundaryUReal = 0.0;
    scalar outputBoundaryUImag = 0.0;

    if (printThisApplication || controls_.abortOnConstraintViolation)
    {
        // div(fvc::flux(U)) is reported but never used as an abort criterion,
        // so its two extra flux+divergence evaluations are only paid for when
        // the diagnostics are actually printed.
        scalar out[6] =
        {
            localMaxAbsDivPhi(arpackOutput_.real.phi),
            localMaxAbsDivPhi(arpackOutput_.imag.phi),
            printThisApplication
              ? localMaxAbsDivFluxOfU(arpackOutput_.real.U) : scalar(0),
            printThisApplication
              ? localMaxAbsDivFluxOfU(arpackOutput_.imag.U) : scalar(0),
            localMaxBoundaryMagU(arpackOutput_.real.U),
            localMaxBoundaryMagU(arpackOutput_.imag.U)
        };
        globalMaxInPlace(out, 6);

        outputDivPhiReal = out[0];
        outputDivPhiImag = out[1];
        outputDivFluxUReal = out[2];
        outputDivFluxUImag = out[3];
        outputBoundaryUReal = out[4];
        outputBoundaryUImag = out[5];
    }

    // The physical hard output constraints are conservative continuity
    // div(phi')=0 and homogeneous physical velocity BCs.  div(fvc::flux(U'))
    // remains useful as a collocated-grid diagnostic, but it is not used as
    // an abort criterion.
    const bool outputConstraintPass =
        outputDivPhiReal <= controls_.constraintTolerance
     && outputDivPhiImag <= controls_.constraintTolerance
     && outputBoundaryUReal <= controls_.constraintTolerance
     && outputBoundaryUImag <= controls_.constraintTolerance;

    if (!outputConstraintPass && controls_.abortOnConstraintViolation)
    {
        FatalErrorInFunction
            << "Shift-invert output from OP application "
            << nextOperatorApplication
            << " left the required discrete homogeneous solenoidal space." << nl
            << "max|div(phi_real)| = " << outputDivPhiReal << nl
            << "max|div(phi_imag)| = " << outputDivPhiImag << nl
            << "max|div(flux(U_real))| = " << outputDivFluxUReal
            << " [diagnostic only]" << nl
            << "max|div(flux(U_imag))| = " << outputDivFluxUImag
            << " [diagnostic only]" << nl
            << "max|U_real on physical walls| = " << outputBoundaryUReal << nl
            << "max|U_imag on physical walls| = " << outputBoundaryUImag << nl
            << "tolerance = " << controls_.constraintTolerance << nl
            << exit(FatalError);
    }

    packVelocity(arpackOutput_, y);

    ++operatorApplications;
    totalInnerIterations += innerIterations;
    maximumInnerIterations = max(maximumInnerIterations, innerIterations);

    if (Pstream::master() && printThisApplication)
    {
        Info<< "PARPACK OP application " << operatorApplications
            << ": inner FGMRES iterations=" << innerIterations
            << ", initialResidual=" << shiftInvertSolver_.lastInitialResidual()
            << ", finalTrueResidual=" << shiftInvertSolver_.lastTrueResidual()
            << ", target=" << shiftInvertSolver_.lastTarget()
            << ", accumulated inner iterations=" << totalInnerIterations
            << nl;

        Info<< "  Arnoldi constraint diagnostics:"
            << " input max|div(flux(U))|=("
            << inputDivFluxUReal << "," << inputDivFluxUImag
            << ") [diagnostic only]"
            << " input max|U_physicalBoundary|=("
            << inputBoundaryUReal << "," << inputBoundaryUImag << ")"
            << nl
            << "    output max|div(phi)|=("
            << outputDivPhiReal << "," << outputDivPhiImag << ")"
            << " output max|div(flux(U))|=("
            << outputDivFluxUReal << "," << outputDivFluxUImag
            << ") [diagnostic only]"
            << " output max|U_physicalBoundary|=("
            << outputBoundaryUReal << "," << outputBoundaryUImag << ")"
            << "  constraintStatus="
            << ((inputConstraintPass && outputConstraintPass) ? "PASS" : "FAIL")
            << nl;
    }
}

scalar ArpackShiftInvertSolver::relativeTransformedVelocityResidual
(
    const ComplexFlow& Tq,
    const ComplexFlow& q,
    const std::complex<double>& mu
) const
{
    const scalar mur = scalar(mu.real());
    const scalar mui = scalar(mu.imag());

    const vectorField& Tr = Tq.real.U.primitiveField();
    const vectorField& Ti = Tq.imag.U.primitiveField();
    const vectorField& qr = q.real.U.primitiveField();
    const vectorField& qi = q.imag.U.primitiveField();

    scalar sums[3] = {0.0, 0.0, 0.0};
    const scalarField& cellV = mesh_.V();

    forAll(Tr, celli)
    {
        const scalar w2 =
            controls_.volumeWeighting
          ? volumeWeightScale_*cellV[celli]
          : 1.0;

        const vector muQr = mur*qr[celli] - mui*qi[celli];
        const vector muQi = mui*qr[celli] + mur*qi[celli];
        const vector dr = Tr[celli] - muQr;
        const vector di = Ti[celli] - muQi;

        sums[0] += w2*((dr & dr) + (di & di));
        sums[1] += w2*((Tr[celli] & Tr[celli]) + (Ti[celli] & Ti[celli]));
        sums[2] += w2*((qr[celli] & qr[celli]) + (qi[celli] & qi[celli]));
    }

    globalSumInPlace(sums, 3);

    const scalar num = Foam::sqrt(max(sums[0], scalar(0)));
    const scalar normT = Foam::sqrt(max(sums[1], scalar(0)));
    const scalar normQ = Foam::sqrt(max(sums[2], scalar(0)));
    const scalar den = normT + scalar(std::abs(mu))*normQ;

    return num/max(den, SMALL);
}

void ArpackShiftInvertSolver::normalizePhase(ComplexFlow& q)
{
    const vectorField& Ur = q.real.U.primitiveField();
    const vectorField& Ui = q.imag.U.primitiveField();

    scalar localMax = -1.0;
    std::complex<double> localPivot(1.0, 0.0);

    forAll(Ur, celli)
    {
        for (direction cmpt=0; cmpt<vector::nComponents; ++cmpt)
        {
            const std::complex<double> value
            (
                double(Ur[celli][cmpt]),
                double(Ui[celli][cmpt])
            );
            const scalar m = scalar(std::abs(value));
            if (m > localMax)
            {
                localMax = m;
                localPivot = value;
            }
        }
    }

    const scalar globalMax = returnReduce(localMax, maxOp<scalar>());
    if (globalMax <= VSMALL) return;

    const label localOwner =
        mag(localMax - globalMax) <= SMALL*max(globalMax, scalar(1))
      ? Pstream::myProcNo()
      : std::numeric_limits<label>::max();
    const label owner = returnReduce(localOwner, minOp<label>());

    double pivotData[2] = {0.0, 0.0};
    if (Pstream::myProcNo() == owner)
    {
        pivotData[0] = localPivot.real();
        pivotData[1] = localPivot.imag();
    }
    MPI_Bcast(pivotData, 2, MPI_DOUBLE, owner, MPI_COMM_WORLD);

    const std::complex<double> pivot(pivotData[0], pivotData[1]);
    if (std::abs(pivot) <= std::numeric_limits<double>::min()) return;

    const std::complex<double> factor = std::conj(pivot)/std::abs(pivot);
    complexScaleFrom(scaledWork_, factor, q);
    copyComplexFlow(q, scaledWork_);
}

void ArpackShiftInvertSolver::reconstructFullMode
(
    const std::complex<double>* packedVelocity,
    const std::complex<double>& mu,
    ComplexFlow& fullMode,
    label& totalInnerIterations,
    label& maximumInnerIterations
)
{
    unpackVelocity(packedVelocity, velocityMode_);
    normalizePhase(velocityMode_);

    label innerIterations = 0;
    const bool converged = shiftInvertSolver_.applyShiftInvert
    (
        velocityMode_,
        imageMode_,
        innerIterations,
        10,
        true
    );
    totalInnerIterations += innerIterations;
    maximumInnerIterations = max(maximumInnerIterations, innerIterations);

    if (!converged || !shiftInvertSolver_.lastSolveConverged())
    {
        FatalErrorInFunction
            << "Final shift-invert application for eigenmode reconstruction "
            << "did not converge. iterations=" << innerIterations
            << " finalTrueResidual=" << shiftInvertSolver_.lastTrueResidual()
            << " target=" << shiftInvertSolver_.lastTarget()
            << exit(FatalError);
    }

    if (std::abs(mu) <= std::numeric_limits<double>::min())
    {
        FatalErrorInFunction
            << "PARPACK returned a zero shift-invert eigenvalue mu; cannot "
            << "recover lambda or the complete mode."
            << exit(FatalError);
    }

    // The PARPACK vector stores velocity only.  If v is an eigenvector of
    // P_U T_sigma E_U with eigenvalue mu, then y=T_sigma E_U v=mu*q.
    // Scale the ENTIRE returned Flow so the pressure-corrected conservative
    // face flux is preserved together with velocity and pressure.
    complexScaleFrom(fullMode, 1.0/mu, imageMode_);
    setHomogeneousVelocityBoundary(fullMode);
}

scalar ArpackShiftInvertSolver::generalizedEigenResidual
(
    ComplexFlow& fullMode,
    const std::complex<double>& lambda
)
{
    shiftInvertSolver_.applySignedJacobian(fullMode, jacobianWork_);
    shiftInvertSolver_.applyB(fullMode, massWork_);

    copyComplexFlow(residualWork_, jacobianWork_);
    complexAxpy(residualWork_, -lambda, massWork_);

    const scalar numerator = shiftInvertSolver_.norm(residualWork_);
    const scalar denominator =
        shiftInvertSolver_.norm(jacobianWork_)
      + scalar(std::abs(lambda))*shiftInvertSolver_.norm(massWork_);

    return numerator/max(denominator, SMALL);
}

ArpackArnoldiResult ArpackShiftInvertSolver::solve(ComplexFlow& leadingMode)
{
    ArpackArnoldiResult result;

    const label nLocalLabel = localSize();
    const label nGlobalLabel = globalSize();

    if (nLocalLabel < 1)
    {
        FatalErrorInFunction
            << "Every PARPACK rank must own at least one velocity degree of "
            << "freedom. This rank has local size " << nLocalLabel
            << exit(FatalError);
    }
    if (controls_.nev < 1 || controls_.nev >= nGlobalLabel - 1)
    {
        FatalErrorInFunction
            << "arnoldiNev must satisfy 1 <= nev < globalSize-1. nev="
            << controls_.nev << " globalSize=" << nGlobalLabel
            << exit(FatalError);
    }

    label ncvLabel = controls_.ncv;
    if (ncvLabel <= 0)
    {
        ncvLabel = min
        (
            nGlobalLabel,
            max(label(20), 2*controls_.nev + 1)
        );
    }
    ncvLabel = min(ncvLabel, nGlobalLabel);

    if (ncvLabel <= controls_.nev + 1)
    {
        FatalErrorInFunction
            << "arnoldiNcv must be greater than arnoldiNev+1. nev="
            << controls_.nev << " ncv=" << ncvLabel
            << exit(FatalError);
    }

    const a_int nLocal = a_int(nLocalLabel);
    const a_int nev = a_int(controls_.nev);
    const a_int ncv = a_int(ncvLabel);
    const a_int ldv = nLocal;
    const a_int ldz = nLocal;
    const a_int lworkl = ncv*(3*ncv + 5);

    std::vector<std::complex<double>> resid(nLocalLabel);
    std::vector<std::complex<double>> V(std::size_t(ldv)*std::size_t(ncv));
    std::vector<std::complex<double>> workd(3*std::size_t(nLocal));
    std::vector<std::complex<double>> workl
    (
        static_cast<std::size_t>(lworkl)
    );
    std::vector<double> rwork(static_cast<std::size_t>(ncv));

    std::vector<a_int> iparam(11, 0);
    std::vector<a_int> ipntr(14, 0);

    iparam[0] = 1; // exact ARPACK shifts
    iparam[2] = a_int(controls_.maxIterations);
    iparam[3] = 1;
    iparam[6] = 1; // standard eigenproblem for the already transformed OP

    a_int ido = 0;

    // Use a deterministic user-supplied starting vector (info=1).  Before
    // PARPACK sees it, projectStartingVector() applies the actual transformed
    // shift-invert operator once, so the normalized v1 starts in the same
    // discrete homogeneous solenoidal velocity space as all later OP outputs.
    if (!controls_.deterministicStart)
    {
        FatalErrorInFunction
            << "arnoldiDeterministicStart must be true for the configured "
            << "deterministic PARPACK start." << nl
            << exit(FatalError);
    }

    a_int info = 1;
    fillStartingVector(resid.data());
    projectStartingVector(resid.data());

    const arpack::which which = whichOption();
    const MPI_Fint communicator = MPI_Comm_c2f(MPI_COMM_WORLD);

    if (Pstream::master())
    {
        Info<< nl
            << "=== PARPACK complex implicitly restarted Arnoldi ===" << nl
            << "ARPACK problem             : T_sigma v = mu v" << nl
            << "T_sigma                    : P_U (J-sigma B)^(-1) E_U" << nl
            << "local velocity DOFs(rank0) : " << nLocalLabel << nl
            << "global velocity DOFs       : " << nGlobalLabel << nl
            << "nev / ncv                  : " << controls_.nev
            << " / " << ncvLabel << nl
            << "which                      : " << controls_.which << nl
            << "Arnoldi tolerance          : " << controls_.tolerance << nl
            << "Arnoldi max iterations     : " << controls_.maxIterations << nl
            << "volume weighting           : " << controls_.volumeWeighting << nl
            << "leading selection          : " << controls_.leadingSelection << nl
            << "constraint tolerance       : " << controls_.constraintTolerance << nl
            << "constraint hard guard      : "
            << controls_.abortOnConstraintViolation
            << nl << nl;
    }

    const auto startTime = std::chrono::steady_clock::now();

    do
    {
        arpack::naupd
        (
            communicator,
            ido,
            arpack::bmat::identity,
            nLocal,
            which,
            nev,
            double(controls_.tolerance),
            resid.data(),
            ncv,
            V.data(),
            ldv,
            iparam.data(),
            ipntr.data(),
            workd.data(),
            workl.data(),
            lworkl,
            rwork.data(),
            info
        );

        if (ido == -1 || ido == 1)
        {
            const a_int xOffset = ipntr[0] - 1;
            const a_int yOffset = ipntr[1] - 1;

            applyArnoldiOperator
            (
                workd.data() + xOffset,
                workd.data() + yOffset,
                result.operatorApplications,
                result.totalInnerIterations,
                result.maximumInnerIterations
            );
        }
        else if (ido == 2)
        {
            // BMAT='I': y = x.
            const a_int xOffset = ipntr[0] - 1;
            const a_int yOffset = ipntr[1] - 1;
            std::copy_n
            (
                workd.data() + xOffset,
                nLocalLabel,
                workd.data() + yOffset
            );
        }
        else if (ido != 99)
        {
            FatalErrorInFunction
                << "Unexpected PARPACK reverse-communication flag ido="
                << ido << exit(FatalError);
        }
    }
    while (ido != 99);

    result.arpackIterations = label(iparam[2]);
    result.nConverged = label(iparam[4]);

    if (info < 0)
    {
        FatalErrorInFunction
            << "pznaupd failed with info=" << info
            << exit(FatalError);
    }
    if (info > 0)
    {
        WarningInFunction
            << "pznaupd returned info=" << info
            << ". Continuing with " << result.nConverged
            << " converged Ritz values." << nl;
    }
    if (result.nConverged < 1)
    {
        FatalErrorInFunction
            << "PARPACK produced no converged Ritz values."
            << exit(FatalError);
    }

    std::vector<a_int> select(std::size_t(ncv), 0);
    std::vector<std::complex<double>> d(std::size_t(controls_.nev + 1));
    std::vector<std::complex<double>> z
    (
        std::size_t(ldz)*std::size_t(controls_.nev + 1)
    );
    std::vector<std::complex<double>> workev(2*std::size_t(ncv));

    a_int extractionInfo = 0;
    arpack::neupd
    (
        communicator,
        a_int(1),
        arpack::howmny::ritz_vectors,
        select.data(),
        d.data(),
        z.data(),
        ldz,
        std::complex<double>(0.0, 0.0),
        workev.data(),
        arpack::bmat::identity,
        nLocal,
        which,
        nev,
        double(controls_.tolerance),
        resid.data(),
        ncv,
        V.data(),
        ldv,
        iparam.data(),
        ipntr.data(),
        workd.data(),
        workl.data(),
        lworkl,
        rwork.data(),
        extractionInfo
    );

    if (extractionInfo != 0)
    {
        FatalErrorInFunction
            << "pzneupd failed with info=" << extractionInfo
            << exit(FatalError);
    }

    const label usable = min(result.nConverged, controls_.nev);
    result.mu.resize(usable);
    result.lambda.resize(usable);

    const std::complex<double> sigma
    (
        double(shiftInvertSolver_.sigmaReal()),
        double(shiftInvertSolver_.sigmaImag())
    );

    result.selectedIndex = 0;
    for (label i=0; i<usable; ++i)
    {
        result.mu[i] = d[i];
        if (!finiteComplex(result.mu[i]) || std::abs(result.mu[i]) == 0.0)
        {
            result.lambda[i] = std::complex<double>
            (
                std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity()
            );
        }
        else
        {
            // ShiftedMALSolver::applySignedJacobian already represents the
            // physical signed J used by the eigenproblem, hence no additional
            // jacobianSign factor belongs here.
            result.lambda[i] = sigma + 1.0/result.mu[i];
        }

        if (i > 0)
        {
            if (controls_.leadingSelection == "largestRealPart")
            {
                if
                (
                    result.lambda[i].real()
                  > result.lambda[result.selectedIndex].real()
                )
                {
                    result.selectedIndex = i;
                }
            }
            else // closestToShift
            {
                if
                (
                    std::abs(result.mu[i])
                  > std::abs(result.mu[result.selectedIndex])
                )
                {
                    result.selectedIndex = i;
                }
            }
        }
    }

    if (Pstream::master())
    {
        Info<< nl << "Converged shift-invert Ritz pairs:" << nl;
        for (label i=0; i<usable; ++i)
        {
            const double frequency =
                result.lambda[i].imag()/(2.0*std::acos(-1.0));
            Info<< "  pair " << i
                << ": mu=(" << result.mu[i].real()
                << "," << result.mu[i].imag() << ")"
                << "  lambda=(" << result.lambda[i].real()
                << "," << result.lambda[i].imag() << ")"
                << "  growth=" << result.lambda[i].real()
                << "  omega=" << result.lambda[i].imag()
                << "  f=" << frequency;
            if (i == result.selectedIndex) Info<< "  <-- selected";
            Info<< nl;
        }
    }

    const std::complex<double>* selectedVelocity =
        z.data() + std::size_t(result.selectedIndex)*std::size_t(ldz);

    reconstructFullMode
    (
        selectedVelocity,
        result.mu[result.selectedIndex],
        leadingMode,
        result.totalInnerIterations,
        result.maximumInnerIterations
    );

    // imageMode_ and velocityMode_ remain Tq and q after reconstruction.
    result.transformedResidual = relativeTransformedVelocityResidual
    (
        imageMode_,
        velocityMode_,
        result.mu[result.selectedIndex]
    );

    result.generalizedResidual = generalizedEigenResidual
    (
        leadingMode,
        result.lambda[result.selectedIndex]
    );

    const auto endTime = std::chrono::steady_clock::now();
    const scalar localWallTime = scalar
    (
        std::chrono::duration<double>(endTime - startTime).count()
    );
    result.wallTime = returnReduce(localWallTime, maxOp<scalar>());

    result.passed =
        result.transformedResidual
          <= controls_.transformedResidualTolerance
     && result.generalizedResidual
          <= controls_.generalizedResidualTolerance;

    if (Pstream::master())
    {
        const std::complex<double>& mu = result.mu[result.selectedIndex];
        const std::complex<double>& lambda = result.lambda[result.selectedIndex];

        Info<< nl
            << "=== Inverse-Arnoldi verification ===" << nl
            << "selected mu                 = (" << mu.real()
            << "," << mu.imag() << ")" << nl
            << "selected lambda             = (" << lambda.real()
            << "," << lambda.imag() << ")" << nl
            << "growth rate Re(lambda)      = " << lambda.real() << nl
            << "angular frequency Im(lambda)= " << lambda.imag() << nl
            << "frequency Im(lambda)/(2pi)  = "
            << lambda.imag()/(2.0*std::acos(-1.0)) << nl
            << "transformed residual        = "
            << result.transformedResidual << " (tol "
            << controls_.transformedResidualTolerance << ')' << nl
            << "generalized residual        = "
            << result.generalizedResidual << " (tol "
            << controls_.generalizedResidualTolerance << ')' << nl
            << "PARPACK update iterations   = " << result.arpackIterations << nl
            << "PARPACK OP applications     = "
            << result.operatorApplications << nl
            << "inner FGMRES iterations     = "
            << result.totalInnerIterations << nl
            << "maximum FGMRES per OP       = "
            << result.maximumInnerIterations << nl
            << "wall time (max rank)        = " << result.wallTime << " s" << nl
            << "INVERSE ARNOLDI STATUS: "
            << (result.passed ? "PASS" : "FAIL") << nl << nl;
    }

    return result;
}

} // namespace matrixFreeStability
} // namespace Foam
