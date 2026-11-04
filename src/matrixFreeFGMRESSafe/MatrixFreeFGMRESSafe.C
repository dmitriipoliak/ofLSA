#include "MatrixFreeFGMRESSafe.H"
#include <cmath>


namespace Foam
{
namespace matrixFreeNewton
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

MatrixFreeFGMRESSolver::MatrixFreeFGMRESSolver
(
    const label itMax,
    const scalar epsilon,
    const scalar absoluteTolerance,
    const bool verboseDiagnostics
)
:
    itMax_(itMax),
    epsilon_(epsilon),
    absoluteTolerance_(max(absoluteTolerance, SMALL)),
    verboseDiagnostics_(verboseDiagnostics)
{}

scalar MatrixFreeFGMRESSolver::innerProduct(const Flow& A, const Flow& B) const
{
    scalar local = 0.0;

    const vectorField& Au = A.U.primitiveField();
    const vectorField& Bu = B.U.primitiveField();
    forAll(Au, celli)
    {
        local += (Au[celli] & Bu[celli]);
    }

    const scalarField& Ap = A.p.primitiveField();
    const scalarField& Bp = B.p.primitiveField();
    forAll(Ap, celli)
    {
        local += Ap[celli]*Bp[celli];
    }

    return returnReduce(local, sumOp<scalar>());
}

scalar MatrixFreeFGMRESSolver::residualNorm(const Flow& A) const
{
    const scalar v = innerProduct(A, A);
    if (!finiteScalar(v))
    {
        return GREAT;
    }
    return Foam::sqrt(max(v, scalar(0)));
}

bool MatrixFreeFGMRESSolver::solveGMRES
(
    Flow& x,
    const Flow& rhs,
    FGMRESWorkspace& ws,
    label& maxIterUsed,
    const bool useRelativeTolerance,
    const label printInterval
)
{
    const label restart = ws.restart;
    const label printEvery = max(label(1), printInterval);

    auto H = [&](const label row, const label col) -> scalar&
    {
        return ws.H[row + (restart + 1)*col];
    };

    rawMatrixAction(x, ws.AX, "FGMRES_Ax_initial");
    flowEqualsAminusB(ws.W, rhs, ws.AX);

    const scalar initialResidual = residualNorm(ws.W);
    const scalar rawTarget = useRelativeTolerance
      ? epsilon_*max(initialResidual, SMALL)
      : epsilon_;
    const scalar target = max(rawTarget, absoluteTolerance_);

    Info<< "FGMRES: initialResidual=" << initialResidual
        << " target=" << target
        << " rawTarget=" << rawTarget
        << " absFloor=" << absoluteTolerance_
        << " restart=" << restart
        << " maxIter=" << itMax_ << nl;

    if (initialResidual <= target)
    {
        maxIterUsed = 0;
        return true;
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

        scalar beta = residualNorm(ws.W);
        if (!finiteScalar(beta) || beta < VSMALL) break;

        copyFlow(ws.V[0], ws.W);
        scaleFlow(ws.V[0], 1.0/beta);
        ws.g[0] = beta;

        label m = 0;

        for (label j=0; j<restart && globalIter<itMax_; ++j)
        {
            ++globalIter;

            // Flexible right preconditioning:
            //     z_j = M_j^{-1} v_j.
            // Here M_j^{-1} is one five-step MSIMPLER application.
            applyRightPreconditioner(ws.V[j], ws.Z[j], globalIter);

            // Arnoldi vector:
            //     w = A z_j.
            rawMatrixAction(ws.Z[j], ws.W, word(string("FGMRES_AZ_") + name(globalIter)));

            // Modified Gram-Schmidt orthogonalization.
            for (label i=0; i<=j; ++i)
            {
                H(i, j) = innerProduct(ws.W, ws.V[i]);
                if (!finiteScalar(H(i, j)))
                {
                    Info<< "FGMRES: non-finite Hessenberg coefficient, "
                        << "stopping current restart at globalIter=" << globalIter << nl;
                    H(i, j) = 0.0;
                    break;
                }
                axpyFlow(ws.W, -H(i, j), ws.V[i]);
            }

            // Second MGS pass (reorthogonalization).
            //
            // On stretched/ill-conditioned grids a single MGS pass can lose
            // orthogonality enough to degrade restarted GMRES.  Accumulate the
            // second-pass coefficients into the same Hessenberg column so the
            // Arnoldi relation remains consistent.
            for (label i=0; i<=j; ++i)
            {
                const scalar h2 = innerProduct(ws.W, ws.V[i]);
                if (finiteScalar(h2))
                {
                    H(i, j) += h2;
                    axpyFlow(ws.W, -h2, ws.V[i]);
                }
                else
                {
                    Info<< "FGMRES: non-finite reorthogonalization coefficient "
                        << "at globalIter=" << globalIter << nl;
                    break;
                }
            }

            H(j+1, j) = residualNorm(ws.W);
            if (finiteScalar(H(j+1, j)) && H(j+1, j) > VSMALL)
            {
                copyFlow(ws.V[j+1], ws.W);
                scaleFlow(ws.V[j+1], 1.0/H(j+1, j));
            }
            else
            {
                H(j+1, j) = 0.0;
                zeroFlow(ws.V[j+1]);
            }

            // Apply previous Givens rotations to the new Hessenberg column.
            for (label i=0; i<j; ++i)
            {
                const scalar hij = H(i, j);
                const scalar hi1j = H(i+1, j);
                H(i, j)   =  ws.cs[i]*hij + ws.sn[i]*hi1j;
                H(i+1, j) = -ws.sn[i]*hij + ws.cs[i]*hi1j;
            }

            // Build and apply the next Givens rotation.  Use hypot()
            // instead of sqrt(h1*h1+h2*h2) to avoid overflow/underflow in
            // long runs with a weak preconditioner.
            const scalar h1 = H(j, j);
            const scalar h2 = H(j+1, j);
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

            H(j, j) = ws.cs[j]*h1 + ws.sn[j]*h2;
            H(j+1, j) = 0.0;

            const scalar gj = ws.g[j];
            ws.g[j]   =  ws.cs[j]*gj;
            ws.g[j+1] = -ws.sn[j]*gj;

            m = j + 1;
            const scalar estimatedResidual = mag(ws.g[j+1]);

            if (!finiteScalar(estimatedResidual))
            {
                Info<< "FGMRES: non-finite estimated residual, "
                    << "leaving current restart at iter=" << globalIter << nl;
                break;
            }

            if (globalIter == 1 || globalIter % printEvery == 0 || estimatedResidual < target)
            {
                Info<< "FGMRES: iter=" << globalIter
                    << " estimatedResidual=" << estimatedResidual
                    << " target=" << target << nl;
            }

            if (estimatedResidual < target)
            {
                break;
            }
        }

        // Solve the small upper-triangular least-squares problem.
        for (label i=m-1; i>=0; --i)
        {
            scalar s = ws.g[i];
            for (label k=i+1; k<m; ++k)
            {
                s -= H(i, k)*ws.y[k];
            }

            if (!finiteScalar(s) || !finiteScalar(H(i, i)) || mag(H(i, i)) < VSMALL)
            {
                ws.y[i] = 0.0;
            }
            else
            {
                ws.y[i] = s/H(i, i);
                if (!finiteScalar(ws.y[i])) ws.y[i] = 0.0;
            }
        }

        // Right-preconditioned update:
        //     x <- x + sum_i y_i z_i.
        for (label i=0; i<m; ++i)
        {
            if (finiteScalar(ws.y[i]))
            {
                axpyFlow(x, ws.y[i], ws.Z[i]);
            }
        }

        // Recompute the true residual at every restart.  This is essential for
        // a matrix-free/preconditioned nonlinear code; the Hessenberg estimate
        // alone can be misleading if the preconditioner is variable.
        rawMatrixAction(x, ws.AX, word(string("FGMRES_Ax_true_") + name(globalIter)));
        flowEqualsAminusB(ws.W, rhs, ws.AX);
        trueResidual = residualNorm(ws.W);

        Info<< "FGMRES: restart end iter=" << globalIter
            << " trueResidual=" << trueResidual
            << " target=" << target << nl;
    }

    maxIterUsed = globalIter;

    const bool converged =
        finiteScalar(trueResidual) && trueResidual <= target;

    if (!converged)
    {
        Info<< "FGMRES: no convergence in " << globalIter
            << " iterations. trueResidual=" << trueResidual
            << " target=" << target << nl;
    }

    return converged;
}

} // namespace matrixFreeNewton
} // namespace Foam
