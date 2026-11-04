#ifndef ComplexFlow_H
#define ComplexFlow_H

#include "MatrixFreeFlow.H"

namespace Foam
{
namespace matrixFreeStability
{

using namespace matrixFreeNewton;

struct ComplexFlow
{
    Flow real;
    Flow imag;

    ComplexFlow
    (
        const word& name,
        const volVectorField& Utemplate,
        const surfaceScalarField& phitemplate,
        const volScalarField& ptemplate
    )
    :
        real(name + "_real", Utemplate, phitemplate, ptemplate),
        imag(name + "_imag", Utemplate, phitemplate, ptemplate)
    {}
};

inline void zeroComplexFlow(ComplexFlow& a)
{
    zeroFlow(a.real);
    zeroFlow(a.imag);
}

inline void copyComplexFlow(ComplexFlow& dst, const ComplexFlow& src)
{
    copyFlow(dst.real, src.real);
    copyFlow(dst.imag, src.imag);
}

inline void scaleComplexFlow(ComplexFlow& a, const scalar s)
{
    scaleFlow(a.real, s);
    scaleFlow(a.imag, s);
}

inline void axpyComplexFlow
(
    ComplexFlow& y,
    const scalar a,
    const ComplexFlow& x
)
{
    axpyFlow(y.real, a, x.real);
    axpyFlow(y.imag, a, x.imag);
}

inline void complexEqualsAminusB
(
    ComplexFlow& dst,
    const ComplexFlow& A,
    const ComplexFlow& B
)
{
    flowEqualsAminusB(dst.real, A.real, B.real);
    flowEqualsAminusB(dst.imag, A.imag, B.imag);
}

inline void setHomogeneousVelocityBoundary(ComplexFlow& a)
{
    matrixFreeNewton::setHomogeneousVelocityBoundary(a.real);
    matrixFreeNewton::setHomogeneousVelocityBoundary(a.imag);
}

inline void replacePhiWithFluxOfU(ComplexFlow& a)
{
    matrixFreeNewton::replacePhiWithFluxOfU(a.real);
    matrixFreeNewton::replacePhiWithFluxOfU(a.imag);
}

} // namespace matrixFreeStability
} // namespace Foam

#endif
