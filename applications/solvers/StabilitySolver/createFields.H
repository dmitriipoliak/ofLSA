Info<< "Reading converged base U, p and phi at time " << runTime.name() << nl;

volVectorField U0
(
    IOobject("U", runTime.name(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE),
    mesh
);

volScalarField pRead
(
    IOobject("p", runTime.name(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE, false),
    mesh
);

// Keep exactly the same Flow-storage convention as NewtonSolver.
const dimensionSet dimVelocityPotential(0, 2, -1, 0, 0, 0, 0);

volScalarField p0
(
    IOobject("baseP", runTime.name(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
    mesh,
    dimensionedScalar("baseP", dimVelocityPotential, 0.0),
    projectionScalarPatchTypes(pRead)
);
copyInternalScalarValues(p0, pRead);

surfaceScalarField phi0
(
    IOobject("phi", runTime.name(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE),
    mesh
);

tmp<volScalarField> tDivPhiDim = fvc::div(phi0);
const dimensionSet dimDivPhi = tDivPhiDim().dimensions();

// StabilitySolver owns its physical, mAL, shift-invert and Arnoldi controls.
// Keep them in one dictionary so a stale Newton case cannot silently change
// the Reynolds number or the stability preconditioner.
IOdictionary stabilityDict
(
    IOobject
    (
        "stabilityProperties",
        runTime.system(),
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
    )
);

const scalar Reynolds = readScalar(stabilityDict.lookup("Re"));
const scalar nuInput = stabilityDict.lookupOrDefault<scalar>("nuInput", 1.0);
if (Reynolds <= 0.0)
{
    FatalErrorInFunction << "Re in stabilityProperties must be positive"
        << exit(FatalError);
}

const scalar nuValue = nuInput/Reynolds;
dimensionedScalar nu
(
    "nu",
    dimensionSet(0, 2, -1, 0, 0, 0, 0),
    nuValue
);

const scalar jacobianSign =
    stabilityDict.lookupOrDefault<scalar>("jacobianSign", -1.0);
const scalar sigmaReal =
    stabilityDict.lookupOrDefault<scalar>("sigmaReal", 0.0);
const scalar sigmaImag =
    stabilityDict.lookupOrDefault<scalar>("sigmaImag", 0.5);
const scalar continuityWeight =
    stabilityDict.lookupOrDefault<scalar>("continuityWeight", 1.0);

const label shiftInvertMaxIter =
    stabilityDict.lookupOrDefault<label>("shiftInvertMaxIter", 300);
const label shiftInvertRestart =
    stabilityDict.lookupOrDefault<label>("shiftInvertRestart", 100);
const scalar shiftInvertTolerance =
    stabilityDict.lookupOrDefault<scalar>("shiftInvertTolerance", 1e-4);
const scalar shiftInvertAbsTolerance =
    stabilityDict.lookupOrDefault<scalar>("shiftInvertAbsTolerance", 1e-10);
const label shiftInvertPrintInterval =
    stabilityDict.lookupOrDefault<label>("shiftInvertPrintInterval", 10);

const scalar baseContinuityTolerance =
    stabilityDict.lookupOrDefault<scalar>("baseContinuityTolerance", 1e-8);
const label arnoldiNev =
    stabilityDict.lookupOrDefault<label>("arnoldiNev", 1);
const label arnoldiNcv =
    stabilityDict.lookupOrDefault<label>("arnoldiNcv", 8);
const label arnoldiMaxIterations =
    stabilityDict.lookupOrDefault<label>("arnoldiMaxIterations", 300);
const scalar arnoldiTolerance =
    stabilityDict.lookupOrDefault<scalar>("arnoldiTolerance", 1.0e-7);
const word arnoldiWhich =
    stabilityDict.lookupOrDefault<word>("arnoldiWhich", "LM");
const bool arnoldiDeterministicStart = bool
(
    stabilityDict.lookupOrDefault<Switch>("arnoldiDeterministicStart", true)
);
const bool arnoldiVolumeWeighting = bool
(
    stabilityDict.lookupOrDefault<Switch>("arnoldiVolumeWeighting", true)
);
const label arnoldiPrintInterval =
    stabilityDict.lookupOrDefault<label>("arnoldiPrintInterval", 1);
const word leadingSelection =
    stabilityDict.lookupOrDefault<word>
    (
        "leadingSelection",
        "largestRealPart"
    );
const scalar arnoldiTransformedResidualTolerance =
    stabilityDict.lookupOrDefault<scalar>
    (
        "arnoldiTransformedResidualTolerance",
        1.0e-6
    );
const scalar arnoldiGeneralizedResidualTolerance =
    stabilityDict.lookupOrDefault<scalar>
    (
        "arnoldiGeneralizedResidualTolerance",
        1.0e-5
    );
const scalar arnoldiConstraintTolerance =
    stabilityDict.lookupOrDefault<scalar>
    (
        "arnoldiConstraintTolerance",
        1.0e-7
    );
const bool arnoldiAbortOnConstraintViolation = bool
(
    stabilityDict.lookupOrDefault<Switch>
    (
        "arnoldiAbortOnConstraintViolation",
        true
    )
);
const Switch writeEigenModeFields
(
    stabilityDict.lookupOrDefault<Switch>("writeEigenModeFields", true)
);

if (!stabilityDict.found("mALCoeffs"))
{
    FatalErrorInFunction
        << "Missing mALCoeffs in system/stabilityProperties"
        << exit(FatalError);
}

const dictionary& malCoeffs = stabilityDict.subDict("mALCoeffs");
const scalar malGamma = readScalar(malCoeffs.lookup("gamma"));
const label malVelocityBlockMaxIter =
    malCoeffs.lookupOrDefault<label>("velocityBlockMaxIter", 50);
const label malVelocityBlockRestart =
    malCoeffs.lookupOrDefault<label>("velocityBlockRestart", 50);
const scalar malVelocityBlockTolerance =
    malCoeffs.lookupOrDefault<scalar>("velocityBlockTolerance", 0.1);
const scalar malVelocityBlockAbsTolerance =
    malCoeffs.lookupOrDefault<scalar>("velocityBlockAbsTolerance", 1e-10);
const label malVelocityBlockPrintInterval =
    malCoeffs.lookupOrDefault<label>("velocityBlockPrintInterval", 10);
const label malOseenShiftSweeps =
    malCoeffs.lookupOrDefault<label>("velocityOseenShiftSweeps", 1);
const Switch malRequireVelocityBlockConvergence
(
    malCoeffs.lookupOrDefault<Switch>("requireVelocityBlockConvergence", false)
);
// Per-iteration reporting of the inner Ux/Uy/Uz solves.  Off by default: it
// prints once per velocityBlockPrintInterval inner iterations, of which there
// are three per mAL application.
const Switch malReportVelocityBlock
(
    malCoeffs.lookupOrDefault<Switch>("reportVelocityBlock", false)
);

if
(
    shiftInvertMaxIter < 1
 || shiftInvertRestart < 1
 || shiftInvertTolerance <= 0.0
 || shiftInvertAbsTolerance <= 0.0
 || continuityWeight <= 0.0
 || baseContinuityTolerance <= 0.0
 || malGamma < 0.0
 || malVelocityBlockMaxIter < 1
 || malVelocityBlockRestart < 1
 || malVelocityBlockTolerance <= 0.0
 || malVelocityBlockAbsTolerance <= 0.0
 || malOseenShiftSweeps < 1
)
{
    FatalErrorInFunction << "Invalid stability/mAL controls" << exit(FatalError);
}

if
(
    arnoldiNev < 1
 || arnoldiNcv <= arnoldiNev + 1
 || arnoldiMaxIterations < 1
 || arnoldiTolerance <= 0.0
 || arnoldiTransformedResidualTolerance <= 0.0
 || arnoldiGeneralizedResidualTolerance <= 0.0
 || arnoldiConstraintTolerance <= 0.0
 || arnoldiPrintInterval < 0
 || (leadingSelection != "largestRealPart" && leadingSelection != "closestToShift")
)
{
    FatalErrorInFunction << "Invalid PARPACK/Arnoldi controls" << exit(FatalError);
}

const label pRefCell = localPressureReferenceCell(mesh, stabilityDict);
const scalar pRefValue =
    stabilityDict.lookupOrDefault<scalar>("pRefValue", 0.0);

volScalarField continuityWork
(
    IOobject("stabilityContinuityWork", runTime.name(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
    mesh,
    dimensionedScalar("stabilityContinuityWork", dimDivPhi, 0.0),
    projectionScalarPatchTypes(pRead)
);
zeroInternalAndCorrectLocal(continuityWork);

surfaceVectorField baseUFaceFrozen
(
    IOobject("stabilityBaseUFaceFrozen", runTime.name(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
    fvc::interpolate(U0)
);

surfaceVectorField momentumFluxWork
(
    IOobject("stabilityMomentumFluxWork", runTime.name(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
    phi0*baseUFaceFrozen
);

// Pressure-Laplacian work fields used by the shifted mAL Schur approximation.
// The L^{-1} action is approximated by one native OpenFOAM GAMG cycle.
volScalarField mALPressurePoisson
(
    IOobject("mALPressurePoisson", runTime.name(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
    mesh,
    dimensionedScalar("mALPressurePoisson", dimVelocityPotential, 0.0),
    projectionScalarPatchTypes(pRead)
);
zeroVolScalar(mALPressurePoisson);

volScalarField mALPressurePoissonStore
(
    IOobject("mALPressurePoissonStore", runTime.name(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
    mesh,
    dimensionedScalar("mALPressurePoissonStore", dimVelocityPotential, 0.0),
    projectionScalarPatchTypes(pRead)
);
zeroVolScalar(mALPressurePoissonStore);

Flow base("StabilityBase", U0, phi0, p0);
setBaseVelocityBoundary(base);
