#ifndef ArpackShiftInvertSolver_H
#define ArpackShiftInvertSolver_H

#include "ShiftedMALSolver.H"
#include "ArpackCompatibility.H"

#include <complex>
#include <vector>

namespace Foam
{
namespace matrixFreeStability
{

struct ArpackArnoldiControls
{
    label nev{1};
    label ncv{8};
    label maxIterations{300};
    scalar tolerance{1.0e-7};
    word which{"LM"};
    bool deterministicStart{true};
    bool volumeWeighting{true};
    label printInterval{1};
    label innerPrintInterval{10};
    word leadingSelection{"largestRealPart"};
    scalar transformedResidualTolerance{1.0e-6};
    scalar generalizedResidualTolerance{1.0e-5};
    // Hard guard for the discrete solenoidal/homogeneous Arnoldi space.
    // Both real and imaginary velocity parts are checked after packing/
    // normalization/unpacking of the start vector and at every OP input/output.
    scalar constraintTolerance{1.0e-7};
    bool abortOnConstraintViolation{true};
};

struct ArpackArnoldiResult
{
    std::vector<std::complex<double>> mu;
    std::vector<std::complex<double>> lambda;

    label selectedIndex{0};
    label nConverged{0};
    label arpackIterations{0};
    label operatorApplications{0};
    label totalInnerIterations{0};
    label maximumInnerIterations{0};

    scalar transformedResidual{GREAT};
    scalar generalizedResidual{GREAT};
    scalar wallTime{0.0};
    bool passed{false};
};

class ArpackShiftInvertSolver
{
    const fvMesh& mesh_;
    ShiftedMALSolver& shiftInvertSolver_;
    ArpackArnoldiControls controls_;
    scalar volumeWeightScale_;

    // sqrt(V_i/Vmean) and its reciprocal.  Both are run constants and are used
    // three times per cell on every pack/unpack, so they are tabulated once
    // instead of recomputing a square root per velocity component.
    scalarField cellWeight_;
    scalarField invCellWeight_;

    ComplexFlow arpackInput_;
    ComplexFlow arpackOutput_;
    ComplexFlow velocityMode_;
    ComplexFlow imageMode_;
    ComplexFlow scaledWork_;
    ComplexFlow jacobianWork_;
    ComplexFlow massWork_;
    ComplexFlow residualWork_;

    label localSize() const;
    label globalSize() const;
    arpack::which whichOption() const;

    void unpackVelocity
    (
        const std::complex<double>* x,
        ComplexFlow& q
    ) const;

    void packVelocity
    (
        const ComplexFlow& q,
        std::complex<double>* x
    ) const;

    void fillStartingVector(std::complex<double>* x);

    void projectStartingVector(std::complex<double>* x);

    void applyArnoldiOperator
    (
        const std::complex<double>* x,
        std::complex<double>* y,
        label& operatorApplications,
        label& totalInnerIterations,
        label& maximumInnerIterations
    );

    scalar relativeTransformedVelocityResidual
    (
        const ComplexFlow& Tq,
        const ComplexFlow& q,
        const std::complex<double>& mu
    ) const;

    void normalizePhase(ComplexFlow& q);

    void reconstructFullMode
    (
        const std::complex<double>* packedVelocity,
        const std::complex<double>& mu,
        ComplexFlow& fullMode,
        label& totalInnerIterations,
        label& maximumInnerIterations
    );

    scalar generalizedEigenResidual
    (
        ComplexFlow& fullMode,
        const std::complex<double>& lambda
    );

public:
    ArpackShiftInvertSolver
    (
        const fvMesh& mesh,
        ShiftedMALSolver& shiftInvertSolver,
        const volVectorField& Utemplate,
        const surfaceScalarField& phitemplate,
        const volScalarField& ptemplate,
        const ArpackArnoldiControls& controls
    );

    ArpackArnoldiResult solve(ComplexFlow& leadingMode);
};

} // namespace matrixFreeStability
} // namespace Foam

#endif
