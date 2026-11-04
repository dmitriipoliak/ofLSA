/*---------------------------------------------------------------------------*\
  Example matrix-vector product for the projected Newton/SIMPLE Krylov action.

  This file is NOT compiled by the library.  It is a template showing how the
  user solver can provide MatrixFreeBiCGStab2::Operator::apply().

  The important point is that Ax.p is NOT zero at the end.  It is the pressure
  correction accumulated by the projection Poisson solve.
\*---------------------------------------------------------------------------*/

#ifndef ProjectedNewtonOperatorSkeleton_H
#define ProjectedNewtonOperatorSkeleton_H

#include "MatrixFreeBiCGStab2.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "fvMatrices.H"
#include "fvc.H"
#include "fvm.H"

namespace Foam
{

class ProjectedNewtonOperatorSkeleton
:
    public MatrixFreeBiCGStab2::Operator
{
    const fvMesh& mesh_;
    const volVectorField& UBase_;
    const dimensionedScalar& nuEff_;  // nu/Re

    // Template pressure-correction field, used for dimensions/BCs.
    const volScalarField& pTemplate_;

public:

    ProjectedNewtonOperatorSkeleton
    (
        const fvMesh& mesh,
        const volVectorField& UBase,
        const dimensionedScalar& nuEff,
        const volScalarField& pTemplate
    )
    :
        mesh_(mesh),
        UBase_(UBase),
        nuEff_(nuEff),
        pTemplate_(pTemplate)
    {}

    virtual void apply
    (
        const MatrixFreeBiCGStab2::VpField& x,
        MatrixFreeBiCGStab2::VpField& Ax
    ) const
    {
        const volVectorField& Utag = x.U;
        const volScalarField& ptag = x.p;

        surfaceScalarField phiBase("phiBase_for_Ax", fvc::flux(UBase_));
        surfaceScalarField phiTag("phiTag_for_Ax", fvc::flux(Utag));

        // Raw linearized momentum action:
        // A Utag + G ptag.
        Ax.U =
            fvc::div(phiBase, Utag)
          + fvc::div(phiTag, UBase_)
          + fvc::grad(ptag)
          - fvc::laplacian(nuEff_, Utag);

        Ax.U.correctBoundaryConditions();

        // Start pressure output from zero.  Do not leave it zero at the end.
        Ax.p = dimensionedScalar("zero", Ax.p.dimensions(), 0.0);

        // SIMPLE/Chorin projection: solve laplacian(phiCorr)=div(Ax.U)
        // and accumulate p-output with phiCorr.
        volScalarField phiCorr
        (
            IOobject
            (
                "phiCorr_Ax",
                mesh_.time().name(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            pTemplate_
        );
        phiCorr = dimensionedScalar("zero", phiCorr.dimensions(), 0.0);

        volScalarField divAxU
        (
            IOobject
            (
                "divAxU",
                mesh_.time().name(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            fvc::div(fvc::flux(Ax.U))
        );

        fvScalarMatrix phiEqn
        (
            fvm::laplacian(phiCorr) == divAxU
        );

        if (phiCorr.needReference())
        {
            phiEqn.setReference(0, 0.0);
        }

        phiEqn.solve();
        phiCorr.correctBoundaryConditions();

        Ax.U -= fvc::grad(phiCorr);
        Ax.U.correctBoundaryConditions();

        // This is the pressure part returned to BiCGStab(2).
        Ax.p += phiCorr;
        Ax.p.correctBoundaryConditions();
    }
};

} // End namespace Foam

#endif

// ************************************************************************* //
