Info<< "Reading U and p; reading phi if present, otherwise constructing phi from U" << nl;

volVectorField U0
(
    IOobject("U", runTime.name(), mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE),
    mesh
);

volScalarField pRead
(
    IOobject("p", runTime.name(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE, false),
    mesh
);

const dimensionSet dimVelocityPotential(0, 2, -1, 0, 0, 0, 0);

volScalarField p0
(
    IOobject("p", runTime.name(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE),
    mesh,
    dimensionedScalar("p", dimVelocityPotential, 0.0),
    projectionScalarPatchTypes(pRead)
);
copyInternalScalarValues(p0, pRead);

// On a restart on the same mesh, preserve the already pressure-corrected phi.
// On a new/mapped mesh, simply omit 0/phi and initialize it from the mapped U.
surfaceScalarField phi0
(
    IOobject
    (
        "phi",
        runTime.name(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    fvc::flux(U0)
);

tmp<volScalarField> tDivPhiDim = fvc::div(phi0);
const dimensionSet dimDivPhi = tDivPhiDim().dimensions();

IOdictionary newtonDict
(
    IOobject
    (
        "newtonProperties",
        runTime.system(),
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
    )
);

const scalar Reynolds = readScalar(newtonDict.lookup("Re"));
const scalar nuInput = newtonDict.lookupOrDefault<scalar>("nuInput", 1.0);
const scalar nuValue = nuInput/Reynolds;

const label maxNewtonIter =
    newtonDict.lookupOrDefault<label>("maxNewtonIter", 30);
const scalar newtonTolerance =
    newtonDict.lookupOrDefault<scalar>("newtonTolerance", 1e-7);
const scalar newtonContinuityTolerance =
    newtonDict.lookupOrDefault<scalar>("newtonContinuityTolerance", 1e-9);
const scalar newtonDamping =
    newtonDict.lookupOrDefault<scalar>("newtonDamping", 1.0);

const label gmresMaxIter =
    newtonDict.lookupOrDefault<label>("gmresMaxIter", 200);
const label gmresRestart =
    newtonDict.lookupOrDefault<label>("gmresRestart", 100);
const scalar gmresTolerance =
    newtonDict.lookupOrDefault<scalar>("gmresTolerance", 0.003);
const scalar gmresAbsTolerance =
    newtonDict.lookupOrDefault<scalar>("gmresAbsTolerance", 5e-8);
const label gmresPrintInterval =
    newtonDict.lookupOrDefault<label>("gmresPrintInterval", 20);

// Projection is part of the nonlinear algorithm in this NewtonSolver version.
const scalar projectionTolerance =
    newtonDict.lookupOrDefault<scalar>("projectionTolerance", 1e-10);
const label projectionMaxIter =
    newtonDict.lookupOrDefault<label>("projectionMaxIter", 3);

if (!newtonDict.found("mALCoeffs"))
{
    FatalErrorInFunction
        << "Missing mALCoeffs in system/newtonProperties"
        << exit(FatalError);
}

const dictionary& malCoeffs = newtonDict.subDict("mALCoeffs");
const scalar malGamma = readScalar(malCoeffs.lookup("gamma"));
const label malVelocityBlockMaxIter =
    malCoeffs.lookupOrDefault<label>("velocityBlockMaxIter", 50);
const label malVelocityBlockRestart =
    malCoeffs.lookupOrDefault<label>("velocityBlockRestart", 50);
const scalar malVelocityBlockTolerance =
    malCoeffs.lookupOrDefault<scalar>("velocityBlockTolerance", 0.05);
const scalar malVelocityBlockAbsTolerance =
    malCoeffs.lookupOrDefault<scalar>("velocityBlockAbsTolerance", 1e-10);

if (Reynolds <= 0.0)
{
    FatalErrorInFunction << "Re must be positive" << exit(FatalError);
}
if (maxNewtonIter < 1 || gmresMaxIter < 1 || gmresRestart < 1)
{
    FatalErrorInFunction << "Iteration counts must be >= 1" << exit(FatalError);
}
if
(
    newtonTolerance <= 0.0
 || newtonContinuityTolerance <= 0.0
 || gmresTolerance <= 0.0
 || gmresAbsTolerance <= 0.0
 || projectionTolerance <= 0.0
 || malVelocityBlockTolerance <= 0.0
 || malVelocityBlockAbsTolerance <= 0.0
)
{
    FatalErrorInFunction << "All tolerances must be positive" << exit(FatalError);
}
if (newtonDamping <= 0.0 || newtonDamping > 1.0)
{
    FatalErrorInFunction << "newtonDamping must satisfy 0 < value <= 1"
        << exit(FatalError);
}
if
(
    projectionMaxIter < 1
 || malGamma < 0.0
 || malVelocityBlockMaxIter < 1
 || malVelocityBlockRestart < 1
)
{
    FatalErrorInFunction << "Invalid projection/mAL controls" << exit(FatalError);
}

const label pRefCell = localPressureReferenceCell(mesh, newtonDict);
const scalar pRefValue =
    newtonDict.lookupOrDefault<scalar>("pRefValue", 0.0);

dimensionedScalar nu
(
    "nu",
    dimensionSet(0, 2, -1, 0, 0, 0, 0),
    nuValue
);

volScalarField rhsContinuity
(
    IOobject("rhsContinuity", runTime.name(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
    mesh,
    dimensionedScalar("rhsContinuity", dimDivPhi, 0.0),
    projectionScalarPatchTypes(pRead)
);
zeroInternalAndCorrect(rhsContinuity);

volScalarField continuityWork
(
    IOobject("continuityWork", runTime.name(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
    mesh,
    dimensionedScalar("continuityWork", dimDivPhi, 0.0),
    projectionScalarPatchTypes(pRead)
);
zeroInternalAndCorrect(continuityWork);

// Work field for the post-update incompressibility projection.
volScalarField pProjection
(
    IOobject("pProjection", runTime.name(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
    mesh,
    dimensionedScalar("pProjection", dimVelocityPotential, 0.0),
    projectionScalarPatchTypes(pRead)
);
zeroVolScalar(pProjection);

surfaceVectorField baseUFaceFrozen
(
    IOobject("baseUFaceFrozen", runTime.name(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
    fvc::interpolate(U0)
);

surfaceVectorField momentumFluxWork
(
    IOobject("momentumFluxWork", runTime.name(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
    phi0*baseUFaceFrozen
);

Flow base("Base", U0, phi0, p0);
setBaseVelocityBoundary(base);
enforcePhysicalPhiFromU(base);

Flow rhsNewton("RHS_Newton", U0, phi0, p0);
Flow newtonCorrection("NewtonCorrection", U0, phi0, p0);
NewtonStateWorkspace gmresWorkspace("NewtonGMRES", U0, phi0, p0, gmresRestart);
