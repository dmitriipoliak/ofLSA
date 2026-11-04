#ifndef MatrixFreeFlow_H
#define MatrixFreeFlow_H

#include "fvMesh.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "fvcDiv.H"
#include "fvcFlux.H"
#include "Pstream.H"
#include "zeroGradientFvPatchFields.H"

namespace Foam
{
namespace matrixFreeNewton
{

struct Flow
{
    volVectorField U;
    surfaceScalarField phi;
    volScalarField p;

    Flow
    (
        const word& name,
        const volVectorField& Utemplate,
        const surfaceScalarField& phitemplate,
        const volScalarField& ptemplate
    )
    :
        U
        (
            IOobject
            (
                word(name + "_U"),
                Utemplate.time().name(),
                Utemplate.db(),
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            Utemplate
        ),
        phi
        (
            IOobject
            (
                word(name + "_phi"),
                phitemplate.time().name(),
                phitemplate.db(),
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            phitemplate
        ),
        p
        (
            IOobject
            (
                word(name + "_p"),
                ptemplate.time().name(),
                ptemplate.db(),
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            ptemplate
        )
    {}
};


inline void zeroVolVector(volVectorField& f)
{
    vectorField& fi = f.primitiveFieldRef();
    forAll(fi, celli) fi[celli] = vector::zero;

    forAll(f.boundaryFieldRef(), patchi)
    {
        if (f.boundaryField()[patchi].coupled()) continue;
        auto& bf = f.boundaryFieldRef()[patchi];
        forAll(bf, i) bf[i] = vector::zero;
    }
    f.correctBoundaryConditions();
}

inline void zeroSurfScalar(surfaceScalarField& f)
{
    scalarField& fi = f.primitiveFieldRef();
    forAll(fi, facei) fi[facei] = 0.0;

    forAll(f.boundaryFieldRef(), patchi)
    {
        auto& bf = f.boundaryFieldRef()[patchi];
        forAll(bf, i) bf[i] = 0.0;
    }
}

inline void zeroVolScalar(volScalarField& f)
{
    scalarField& fi = f.primitiveFieldRef();
    forAll(fi, celli) fi[celli] = 0.0;

    forAll(f.boundaryFieldRef(), patchi)
    {
        if (f.boundaryField()[patchi].coupled()) continue;
        auto& bf = f.boundaryFieldRef()[patchi];
        forAll(bf, i) bf[i] = 0.0;
    }
    f.correctBoundaryConditions();
}

inline void zeroFlow(Flow& a)
{
    zeroVolVector(a.U);
    zeroSurfScalar(a.phi);
    zeroVolScalar(a.p);
}

inline void copyVolVector(volVectorField& dst, const volVectorField& src)
{
    vectorField& dI = dst.primitiveFieldRef();
    const vectorField& sI = src.primitiveField();
    forAll(dI, celli) dI[celli] = sI[celli];

    forAll(dst.boundaryFieldRef(), patchi)
    {
        auto& d = dst.boundaryFieldRef()[patchi];
        const auto& s = src.boundaryField()[patchi];
        forAll(d, i) d[i] = s[i];
    }
}

inline void copySurfScalar(surfaceScalarField& dst, const surfaceScalarField& src)
{
    scalarField& dI = dst.primitiveFieldRef();
    const scalarField& sI = src.primitiveField();
    forAll(dI, facei) dI[facei] = sI[facei];

    forAll(dst.boundaryFieldRef(), patchi)
    {
        auto& d = dst.boundaryFieldRef()[patchi];
        const auto& s = src.boundaryField()[patchi];
        forAll(d, i) d[i] = s[i];
    }
}

inline void copyVolScalar(volScalarField& dst, const volScalarField& src)
{
    scalarField& dI = dst.primitiveFieldRef();
    const scalarField& sI = src.primitiveField();
    forAll(dI, celli) dI[celli] = sI[celli];

    forAll(dst.boundaryFieldRef(), patchi)
    {
        auto& d = dst.boundaryFieldRef()[patchi];
        const auto& s = src.boundaryField()[patchi];
        forAll(d, i) d[i] = s[i];
    }
}

inline void copyFlow(Flow& dst, const Flow& src)
{
    copyVolVector(dst.U, src.U);
    copySurfScalar(dst.phi, src.phi);
    copyVolScalar(dst.p, src.p);
}

inline void axpyVolVector(volVectorField& y, const scalar a, const volVectorField& x)
{
    vectorField& yI = y.primitiveFieldRef();
    const vectorField& xI = x.primitiveField();
    forAll(yI, celli) yI[celli] += a*xI[celli];

    forAll(y.boundaryFieldRef(), patchi)
    {
        auto& yy = y.boundaryFieldRef()[patchi];
        const auto& xx = x.boundaryField()[patchi];
        forAll(yy, i) yy[i] += a*xx[i];
    }
}

inline void axpySurfScalar(surfaceScalarField& y, const scalar a, const surfaceScalarField& x)
{
    scalarField& yI = y.primitiveFieldRef();
    const scalarField& xI = x.primitiveField();
    forAll(yI, facei) yI[facei] += a*xI[facei];

    forAll(y.boundaryFieldRef(), patchi)
    {
        auto& yy = y.boundaryFieldRef()[patchi];
        const auto& xx = x.boundaryField()[patchi];
        forAll(yy, i) yy[i] += a*xx[i];
    }
}

inline void axpyVolScalar(volScalarField& y, const scalar a, const volScalarField& x)
{
    scalarField& yI = y.primitiveFieldRef();
    const scalarField& xI = x.primitiveField();
    forAll(yI, celli) yI[celli] += a*xI[celli];

    forAll(y.boundaryFieldRef(), patchi)
    {
        auto& yy = y.boundaryFieldRef()[patchi];
        const auto& xx = x.boundaryField()[patchi];
        forAll(yy, i) yy[i] += a*xx[i];
    }
}

inline void axpyFlow(Flow& y, const scalar a, const Flow& x)
{
    axpyVolVector(y.U, a, x.U);
    axpySurfScalar(y.phi, a, x.phi);
    axpyVolScalar(y.p, a, x.p);
}

inline void scaleFlow(Flow& y, const scalar a)
{
    vectorField& yU = y.U.primitiveFieldRef();
    forAll(yU, celli) yU[celli] *= a;

    scalarField& yPhi = y.phi.primitiveFieldRef();
    forAll(yPhi, facei) yPhi[facei] *= a;

    scalarField& yP = y.p.primitiveFieldRef();
    forAll(yP, celli) yP[celli] *= a;

    forAll(y.U.boundaryFieldRef(), patchi)
    {
        auto& Ubf = y.U.boundaryFieldRef()[patchi];
        forAll(Ubf, i) Ubf[i] *= a;

        auto& phibf = y.phi.boundaryFieldRef()[patchi];
        forAll(phibf, i) phibf[i] *= a;

        auto& pbf = y.p.boundaryFieldRef()[patchi];
        forAll(pbf, i) pbf[i] *= a;
    }
}

inline void flowEqualsAminusB(Flow& dst, const Flow& A, const Flow& B)
{
    if (&dst == &A && &dst == &B)
    {
        zeroFlow(dst);
    }
    else if (&dst == &A)
    {
        axpyFlow(dst, -1.0, B);
    }
    else if (&dst == &B)
    {
        scaleFlow(dst, -1.0);
        axpyFlow(dst, 1.0, A);
    }
    else
    {
        copyFlow(dst, A);
        axpyFlow(dst, -1.0, B);
    }
}

inline void flowEqualsAminusBetaB(Flow& dst, const Flow& A, const scalar beta, const Flow& B)
{
    if (&dst == &A && &dst == &B)
    {
        scaleFlow(dst, 1.0 - beta);
    }
    else if (&dst == &A)
    {
        axpyFlow(dst, -beta, B);
    }
    else if (&dst == &B)
    {
        scaleFlow(dst, -beta);
        axpyFlow(dst, 1.0, A);
    }
    else
    {
        copyFlow(dst, A);
        axpyFlow(dst, -beta, B);
    }
}

inline void addFromTmp(volVectorField& dst, const tmp<volVectorField>& t, const scalar a = 1.0)
{
    const volVectorField& src = t();
    vectorField& dI = dst.primitiveFieldRef();
    const vectorField& sI = src.primitiveField();
    forAll(dI, celli) dI[celli] += a*sI[celli];

    forAll(dst.boundaryFieldRef(), patchi)
    {
        auto& d = dst.boundaryFieldRef()[patchi];
        const auto& s = src.boundaryField()[patchi];
        forAll(d, i) d[i] += a*s[i];
    }
}

inline void replacePhiWithFluxOfU(Flow& a)
{
    tmp<surfaceScalarField> tphi = fvc::flux(a.U);
    copySurfScalar(a.phi, tphi());
}

inline scalar maxMagU(const Flow& a)
{
    scalar m = 0.0;
    const vectorField& Ui = a.U.primitiveField();
    forAll(Ui, celli)
    {
        m = max(m, mag(Ui[celli].x()));
        m = max(m, mag(Ui[celli].y()));
        m = max(m, mag(Ui[celli].z()));
    }
    return returnReduce(m, maxOp<scalar>());
}

inline scalar maxMagP(const Flow& a)
{
    scalar m = 0.0;
    const scalarField& pi = a.p.primitiveField();
    forAll(pi, celli) m = max(m, mag(pi[celli]));
    return returnReduce(m, maxOp<scalar>());
}

inline scalar maxFlow(const Flow& a)
{
    return maxMagU(a);
}

inline scalar maxAbsDivPhi(const surfaceScalarField& phi)
{
    tmp<volScalarField> tdiv = fvc::div(phi);
    return gMax(mag(tdiv().primitiveField()));
}

inline scalar innerProductFlow(const Flow& A, const Flow& B)
{
    scalar local = 0.0;
    const vectorField& Au = A.U.primitiveField();
    const vectorField& Bu = B.U.primitiveField();
    forAll(Au, celli)
    {
        local += (Au[celli] & Bu[celli]);
    }
    return returnReduce(local, sumOp<scalar>());
}

inline scalar normUFlow(const Flow& A)
{
    scalar v = innerProductFlow(A, A);
    if (v < 0) v = 0;
    return Foam::sqrt(v);
}

inline scalar maxPhiMag(const Flow& A)
{
    scalar m = 0.0;
    const scalarField& phii = A.phi.primitiveField();
    forAll(phii, facei) m = max(m, mag(phii[facei]));
    forAll(A.phi.boundaryField(), patchi)
    {
        if (A.phi.boundaryField()[patchi].coupled()) continue;
        const auto& phip = A.phi.boundaryField()[patchi];
        forAll(phip, i) m = max(m, mag(phip[i]));
    }
    return returnReduce(m, maxOp<scalar>());
}

inline void printFlowDiag(const string& name, const Flow& A)
{
    Info<< "    diag[" << name << "]: "
        << "maxU=" << maxMagU(A)
        << " normU=" << normUFlow(A)
        << " maxPhi=" << maxPhiMag(A)
        << " divPhi=" << maxAbsDivPhi(A.phi)
        << " maxP=" << maxMagP(A)
        << nl;
}

inline scalar distanceFlow(const Flow& A, const Flow& B)
{
    scalar d = 0.0;
    const vectorField& Au = A.U.primitiveField();
    const vectorField& Bu = B.U.primitiveField();
    forAll(Au, celli)
    {
        scalar denom = mag(Au[celli]);
        scalar val = mag(Au[celli] - Bu[celli]);
        if (denom > 1.0) val /= denom;
        d = max(d, val);
    }
    return returnReduce(d, maxOp<scalar>());
}

inline scalar cnormbComponent(const Flow& correction, const Flow& base, const direction cmpt)
{
    scalar m = 0.0;
    const vectorField& c = correction.U.primitiveField();
    const vectorField& b = base.U.primitiveField();
    forAll(c, celli)
    {
        scalar val = mag(c[celli].component(cmpt));
        scalar den = mag(b[celli].component(cmpt));
        if (den > 1.0) val /= den;
        m = max(m, val);
    }
    return returnReduce(m, maxOp<scalar>());
}

inline void setHomogeneousVelocityBoundary(Flow& a)
{
    forAll(a.U.boundaryFieldRef(), patchi)
    {
        if (a.U.boundaryField()[patchi].coupled()) continue;
        auto& Ubf = a.U.boundaryFieldRef()[patchi];
        forAll(Ubf, i) Ubf[i] = vector::zero;
    }

    forAll(a.phi.boundaryFieldRef(), patchi)
    {
        if (a.phi.boundaryField()[patchi].coupled()) continue;
        auto& phibf = a.phi.boundaryFieldRef()[patchi];
        forAll(phibf, i) phibf[i] = 0.0;
    }
    a.U.correctBoundaryConditions();
}

inline void setBaseVelocityBoundary(Flow& base)
{
    base.U.correctBoundaryConditions();
    const surfaceVectorField& Sf = base.U.mesh().Sf();
    forAll(base.phi.boundaryFieldRef(), patchi)
    {
        if (base.phi.boundaryField()[patchi].coupled()) continue;
        auto& phibf = base.phi.boundaryFieldRef()[patchi];
        const auto& Ubf = base.U.boundaryField()[patchi];
        const auto& Sfbf = Sf.boundaryField()[patchi];
        forAll(phibf, i) phibf[i] = (Ubf[i] & Sfbf[i]);
    }
}

inline wordList projectionScalarPatchTypes(const volScalarField& pTemplate)
{
    const fvMesh& mesh = pTemplate.mesh();
    wordList patchTypes(mesh.boundary().size());
    forAll(mesh.boundary(), patchi)
    {
        if (mesh.boundary()[patchi].coupled())
        {
            patchTypes[patchi] = pTemplate.boundaryField()[patchi].type();
        }
        else
        {
            patchTypes[patchi] = zeroGradientFvPatchScalarField::typeName;
        }
    }
    return patchTypes;
}

inline void copyInternalScalarValues(volScalarField& dst, const volScalarField& src)
{
    scalarField& dI = dst.primitiveFieldRef();
    const scalarField& sI = src.primitiveField();
    forAll(dI, celli) dI[celli] = sI[celli];

    forAll(dst.boundaryFieldRef(), patchi)
    {
        if (dst.boundaryField()[patchi].coupled()) continue;
        auto& d = dst.boundaryFieldRef()[patchi];
        const auto& s = src.boundaryField()[patchi];
        forAll(d, i) d[i] = s[i];
    }
    dst.correctBoundaryConditions();
}

} // namespace matrixFreeNewton
} // namespace Foam

#endif
