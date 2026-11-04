#ifndef ArpackCompatibility_H
#define ArpackCompatibility_H

// ARPACK-NG installs the ISO_C_BINDING C++ wrapper either directly under the
// include prefix or under an arpack/ subdirectory, depending on the package.
#if defined(__has_include)
# if __has_include(<arpack/parpack.hpp>)
#  include <arpack/parpack.hpp>
# elif __has_include(<parpack.hpp>)
#  include <parpack.hpp>
# else
#  error "PARPACK C++ header not found. Install ARPACK-NG with MPI and ISO_C_BINDING development headers (parpack.hpp)."
# endif
#else
# include <parpack.hpp>
#endif

#endif
