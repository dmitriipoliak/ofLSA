#include "DoubledRealFGMRES.H"
#include "Pstream.H"
#include <mpi.h>
#include <cmath>

namespace Foam
{
namespace matrixFreeStability
{

namespace
{
inline bool finiteScalar(const scalar x)
{
    return std::isfinite(static_cast<double>(x));
}

inline scalar robustHypot(const scalar a, const scalar b)
{
    return scalar(std::hypot(static_cast<double>(a), static_cast<double>(b)));
}
}

void globalSumInPlace(scalar* values, const label n)
{
    static_assert
    (
        sizeof(scalar) == sizeof(double),
        "The batched Krylov reduction requires double precision (WM_PRECISION_OPTION=DP)"
    );

    if (n <= 0 || !Pstream::parRun()) return;

    // MPI_IN_PLACE expands to an old-style cast in the OpenMPI headers.
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wold-style-cast"
    MPI_Allreduce
    (
        MPI_IN_PLACE,
        values,
        int(n),
        MPI_DOUBLE,
        MPI_SUM,
        MPI_COMM_WORLD
    );
    #pragma GCC diagnostic pop
}


DoubledRealKrylovBase::DoubledRealKrylovBase
(
    const fvMesh& mesh,
    const label itMax,
    const scalar epsilon,
    const scalar absoluteTolerance,
    const bool reorthogonalize
)
:
    krylovMesh_(mesh),
    volumeWeightScale_(1.0),
    itMax_(itMax),
    epsilon_(epsilon),
    absoluteTolerance_(max(absoluteTolerance, SMALL)),
    orthogonalizationPasses_(reorthogonalize ? 2 : 1),
    lastSolveConverged_(false),
    lastInitialResidual_(GREAT),
    lastTrueResidual_(GREAT),
    lastTarget_(GREAT),
    lastIterations_(0)
{
    const scalarField& V = krylovMesh_.V();
    scalar localVolume = 0.0;
    forAll(V, celli) localVolume += V[celli];

    scalar reduced[2] = {localVolume, scalar(krylovMesh_.nCells())};
    globalSumInPlace(reduced, 2);

    const scalar globalVolume = reduced[0];
    const scalar globalNCells = reduced[1];

    if
    (
        globalNCells <= 0
     || !finiteScalar(globalVolume)
     || globalVolume <= VSMALL
    )
    {
        FatalErrorInFunction
            << "Cannot construct volume-weighted FGMRES metric"
            << exit(FatalError);
    }

    // w_i = V_i/Vmean.  Uniform meshes therefore retain w_i=1 exactly.
    volumeWeightScale_ = globalNCells/globalVolume;
}


DoubledRealFGMRESSolver::DoubledRealFGMRESSolver
(
    const fvMesh& mesh,
    const label itMax,
    const scalar epsilon,
    const scalar absoluteTolerance,
    const bool reorthogonalize,
    const scalar pressureWeight
)
:
    DoubledRealKrylovBase
    (
        mesh,
        itMax,
        epsilon,
        absoluteTolerance,
        reorthogonalize
    ),
    pressureWeightSquared_(pressureWeight*pressureWeight)
{}


scalar DoubledRealFGMRESSolver::innerProduct
(
    const ComplexStateVector& A,
    const ComplexStateVector& B
) const
{
    const scalarField& V = krylovMesh_.V();
    const scalar w2 = pressureWeightSquared_;

    scalar local = 0.0;
    forAll(A.Ur, celli)
    {
        const scalar w = volumeWeightScale_*V[celli];
        local += w*
        (
            (A.Ur[celli] & B.Ur[celli])
          + (A.Ui[celli] & B.Ui[celli])
          + w2*(A.pr[celli]*B.pr[celli] + A.pi[celli]*B.pi[celli])
        );
    }

    globalSumInPlace(&local, 1);
    return local;
}


scalar DoubledRealFGMRESSolver::residualNorm
(
    const ComplexStateVector& A
) const
{
    const scalar v = innerProduct(A, A);
    if (!finiteScalar(v)) return GREAT;
    return Foam::sqrt(max(v, scalar(0)));
}


scalar DoubledRealFGMRESSolver::flowNorm(const ComplexFlow& A) const
{
    const scalarField& V = krylovMesh_.V();
    const scalar w2 = pressureWeightSquared_;

    const vectorField& ur = A.real.U.primitiveField();
    const vectorField& ui = A.imag.U.primitiveField();
    const scalarField& pr = A.real.p.primitiveField();
    const scalarField& pi = A.imag.p.primitiveField();

    scalar local = 0.0;
    forAll(ur, celli)
    {
        const scalar w = volumeWeightScale_*V[celli];
        local += w*
        (
            (ur[celli] & ur[celli])
          + (ui[celli] & ui[celli])
          + w2*(pr[celli]*pr[celli] + pi[celli]*pi[celli])
        );
    }

    globalSumInPlace(&local, 1);
    if (!finiteScalar(local)) return GREAT;
    return Foam::sqrt(max(local, scalar(0)));
}


void DoubledRealFGMRESSolver::basisInnerProducts
(
    const ComplexStateVector& W,
    const PtrList<ComplexStateVector>& V,
    const label n,
    scalar* dots
) const
{
    const scalarField& cellV = krylovMesh_.V();
    const scalar w2 = pressureWeightSquared_;

    for (label i=0; i<n; ++i)
    {
        const ComplexStateVector& Vi = V[i];

        scalar local = 0.0;
        forAll(W.Ur, celli)
        {
            const scalar w = volumeWeightScale_*cellV[celli];
            local += w*
            (
                (W.Ur[celli] & Vi.Ur[celli])
              + (W.Ui[celli] & Vi.Ui[celli])
              + w2*(W.pr[celli]*Vi.pr[celli] + W.pi[celli]*Vi.pi[celli])
            );
        }

        dots[i] = local;
    }

    // One collective for the whole Gram-Schmidt pass.
    globalSumInPlace(dots, n);
}


void DoubledRealFGMRESSolver::finaliseFlowState(ComplexFlow& x)
{
    // The homogeneous velocity boundary condition and phi=fvc::flux(U) are the
    // exact invariants that the mAL preconditioner output satisfies, so
    // rebuilding them from the internal state reproduces the field the
    // operator would have been given.
    setHomogeneousVelocityBoundary(x);
    x.real.p.correctBoundaryConditions();
    x.imag.p.correctBoundaryConditions();
    replacePhiWithFluxOfU(x);
}


void DoubledRealFGMRESSolver::solveGMRES
(
    ComplexFlow& x,
    const ComplexFlow& rhs,
    ComplexFGMRESWorkspace& ws,
    label& maxIterUsed,
    const bool useRelativeTolerance,
    const label printInterval,
    const bool report,
    const bool zeroInitialGuess
)
{
    const label restart = ws.restart;
    const label printEvery = max(label(1), printInterval);

    auto H = [&](const label row, const label col) -> scalar&
    {
        return ws.H[row + (restart + 1)*col];
    };

    if (zeroInitialGuess)
    {
        // A is linear, so A*0 is exactly zero and r0 is exactly rhs.
        loadComplexState(ws.W, rhs);
    }
    else
    {
        rawMatrixAction(x, ws.AX);
        stateEqualsFlowMinusFlow(ws.W, rhs, ws.AX);
    }

    const scalar initialResidual = residualNorm(ws.W);
    const scalar rawTarget = useRelativeTolerance
      ? epsilon_*max(initialResidual, SMALL)
      : epsilon_;
    const scalar target = max(rawTarget, absoluteTolerance_);

    // Normalize reported residuals by the initial residual of this solve.
    // Convergence still uses the original raw residuals.
    const scalar residualScale =
        (finiteScalar(initialResidual) && initialResidual > VSMALL)
      ? initialResidual
      : scalar(1);
    const scalar normalizedInitialResidual = initialResidual/residualScale;
    const scalar normalizedTarget = target/residualScale;

    lastSolveConverged_ = false;
    lastInitialResidual_ = normalizedInitialResidual;
    lastTrueResidual_ = normalizedInitialResidual;
    lastTarget_ = normalizedTarget;
    lastIterations_ = 0;

    if (report)
    {
        Info<< "FGMRES(shifted): initialResidual=" << normalizedInitialResidual
            << " target=" << normalizedTarget
            << " restart=" << restart
            << " maxIter=" << itMax_ << nl;
    }

    if (initialResidual <= target)
    {
        maxIterUsed = 0;
        lastSolveConverged_ = true;
        return;
    }

    label globalIter = 0;
    scalar trueResidual = initialResidual;

    while (globalIter < itMax_ && trueResidual > target)
    {
        forAll(ws.H, i) ws.H[i] = 0.0;
        forAll(ws.cs, i) ws.cs[i] = 0.0;
        forAll(ws.sn, i) ws.sn[i] = 0.0;
        forAll(ws.g, i) ws.g[i] = 0.0;
        forAll(ws.y, i) ws.y[i] = 0.0;

        const scalar beta = residualNorm(ws.W);
        if (!finiteScalar(beta) || beta < VSMALL) break;

        copyComplexState(ws.V[0], ws.W);
        scaleComplexState(ws.V[0], 1.0/beta);
        ws.g[0] = beta;

        label m = 0;

        for (label j=0; j<restart && globalIter<itMax_; ++j)
        {
            ++globalIter;

            // The mAL preconditioner reads the internal velocity and pressure
            // of its argument only, so no boundary evaluation is needed here.
            storeComplexState(ws.precondInput, ws.V[j]);
            applyRightPreconditioner
            (
                ws.precondInput,
                ws.precondOutput,
                globalIter
            );

            // Keep the compact image for the solution update, then apply the
            // operator to the full field the preconditioner just produced.
            loadComplexState(ws.Z[j], ws.precondOutput);
            rawMatrixAction(ws.precondOutput, ws.AX);
            loadComplexState(ws.W, ws.AX);

            // Gram-Schmidt with one reorthogonalization (CGS2).  Each pass
            // needs exactly one global reduction instead of one per basis
            // vector, which is what makes the orthogonalization scale.
            for (label pass=0; pass<orthogonalizationPasses_; ++pass)
            {
                basisInnerProducts(ws.W, ws.V, j+1, ws.dots.begin());

                for (label i=0; i<=j; ++i)
                {
                    scalar h = ws.dots[i];
                    if (!finiteScalar(h)) h = 0.0;

                    H(i,j) += h;
                    axpyComplexState(ws.W, -h, ws.V[i]);
                }
            }

            H(j+1,j) = residualNorm(ws.W);

            if (finiteScalar(H(j+1,j)) && H(j+1,j) > VSMALL)
            {
                copyComplexState(ws.V[j+1], ws.W);
                scaleComplexState(ws.V[j+1], 1.0/H(j+1,j));
            }
            else
            {
                H(j+1,j) = 0.0;
                zeroComplexState(ws.V[j+1]);
            }

            for (label i=0; i<j; ++i)
            {
                const scalar hij = H(i,j);
                const scalar hi1j = H(i+1,j);
                H(i,j) = ws.cs[i]*hij + ws.sn[i]*hi1j;
                H(i+1,j) = -ws.sn[i]*hij + ws.cs[i]*hi1j;
            }

            const scalar h1 = H(j,j);
            const scalar h2 = H(j+1,j);
            const scalar den = robustHypot(h1, h2);

            if (finiteScalar(den) && den > VSMALL)
            {
                ws.cs[j] = h1/den;
                ws.sn[j] = h2/den;
            }
            else
            {
                ws.cs[j] = 1.0;
                ws.sn[j] = 0.0;
            }

            H(j,j) = ws.cs[j]*h1 + ws.sn[j]*h2;
            H(j+1,j) = 0.0;

            const scalar gj = ws.g[j];
            ws.g[j] = ws.cs[j]*gj;
            ws.g[j+1] = -ws.sn[j]*gj;

            m = j+1;
            const scalar estimatedResidual = mag(ws.g[j+1]);
            const scalar normalizedEstimatedResidual =
                estimatedResidual/residualScale;

            if
            (
                report
             &&
                (
                    globalIter % printEvery == 0
                 || estimatedResidual <= target
                )
            )
            {
                Info<< "FGMRES(shifted): iter=" << globalIter
                    << " estimatedResidual=" << normalizedEstimatedResidual
                    << " target=" << normalizedTarget << nl;
            }

            if (!finiteScalar(estimatedResidual) || estimatedResidual <= target)
            {
                break;
            }
        }

        for (label i=m-1; i>=0; --i)
        {
            scalar s = ws.g[i];
            for (label k=i+1; k<m; ++k) s -= H(i,k)*ws.y[k];

            if
            (
                !finiteScalar(s)
             || !finiteScalar(H(i,i))
             || mag(H(i,i)) < VSMALL
            )
            {
                ws.y[i] = 0.0;
            }
            else
            {
                ws.y[i] = s/H(i,i);
                if (!finiteScalar(ws.y[i])) ws.y[i] = 0.0;
            }
        }

        for (label i=0; i<m; ++i)
        {
            if (finiteScalar(ws.y[i])) axpyStateIntoFlow(x, ws.y[i], ws.Z[i]);
        }

        // x is the only field whose boundary values and conservative flux are
        // consumed downstream, so rebuild them from the updated state.
        finaliseFlowState(x);

        rawMatrixAction(x, ws.AX);
        stateEqualsFlowMinusFlow(ws.W, rhs, ws.AX);
        trueResidual = residualNorm(ws.W);

        if (report)
        {
            Info<< "FGMRES(shifted): restart end iter=" << globalIter
                << " trueResidual=" << trueResidual/residualScale
                << " target=" << normalizedTarget << nl;
        }
    }

    maxIterUsed = globalIter;
    lastIterations_ = globalIter;
    lastTrueResidual_ = trueResidual/residualScale;
    lastSolveConverged_ = finiteScalar(trueResidual) && trueResidual <= target;

    if (report)
    {
        Info<< "FGMRES(shifted): "
            << (lastSolveConverged_ ? "CONVERGED" : "FAILED")
            << " iterations=" << globalIter
            << " trueResidual=" << trueResidual/residualScale
            << " target=" << normalizedTarget << nl;
    }
}

} // namespace matrixFreeStability
} // namespace Foam
