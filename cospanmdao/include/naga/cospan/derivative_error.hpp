#pragma once
#include <stdexcept>
namespace naga::cospan { struct DerivativeError : std::runtime_error { using std::runtime_error::runtime_error; }; }
