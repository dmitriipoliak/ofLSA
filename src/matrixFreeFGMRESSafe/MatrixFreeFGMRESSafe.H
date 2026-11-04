#ifndef MatrixFreeFGMRESSafe_H
#define MatrixFreeFGMRESSafe_H

#include "MatrixFreeFlow.H"
#include "PtrList.H"

namespace Foam
{
namespace matrixFreeNewton
{

struct FGMRESWorkspace
{
    label restart;
    PtrList<Flow> V;
    PtrList<Flow> Z;
    Flow W;
    Flow AX;
    List<scalar> H;
    List<scalar> cs;
    List<scalar> sn;
    List<scalar> g;
    List<scalar> y;

    FGMRESWorkspace
    (
        const word& prefix,
        const volVectorField& Utemplate,
        const surfaceScalarField& phitemplate,
        const volScalarField& ptemplate,
        const label restartSize
    )
    :
        restart(max(label(1), restartSize)),
        V(restart + 1),
        Z(restart),
        W(prefix + "_W", Utemplate, phitemplate, ptemplate),
        AX(prefix + "_AX", Utemplate, phitemplate, ptemplate),
        H((restart + 1)*restart, 0.0),
        cs(restart, 0.0),
        sn(restart, 0.0),
        g(restart + 1, 0.0),
        y(restart, 0.0)
    {
        for (label i=0; i<restart+1; ++i)
        {
            V.set(i, new Flow(prefix + "_V" + name(i), Utemplate, phitemplate, ptemplate));
        }
        for (label i=0; i<restart; ++i)
        {
            Z.set(i, new Flow(prefix + "_Z" + name(i), Utemplate, phitemplate, ptemplate));
        }
    }
};

class MatrixFreeFGMRESSolver
{
protected:
    label itMax_;
    scalar epsilon_;
    scalar absoluteTolerance_;
    bool verboseDiagnostics_;

    virtual void rawMatrixAction(Flow& x, Flow& Ax, const word& labelName) = 0;
    virtual void applyRightPreconditioner(const Flow& r, Flow& z, const label iter) = 0;

    virtual scalar innerProduct(const Flow& A, const Flow& B) const;
    virtual scalar residualNorm(const Flow& A) const;

public:
    MatrixFreeFGMRESSolver
    (
        const label itMax,
        const scalar epsilon,
        const scalar absoluteTolerance,
        const bool verboseDiagnostics
    );

    virtual ~MatrixFreeFGMRESSolver() = default;

    bool solveGMRES
    (
        Flow& x,
        const Flow& rhs,
        FGMRESWorkspace& ws,
        label& maxIterUsed,
        const bool useRelativeTolerance = true,
        const label printInterval = 10
    );
};

} // namespace matrixFreeNewton
} // namespace Foam

#endif
