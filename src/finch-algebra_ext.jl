# This file declares some important algebraic properties of major functions
# so that Finch can produce efficient code.

struct InitMin{D} end
struct InitMax{D} end

(f::InitMin{D})(x) where {D} = x
(f::InitMax{D})(x) where {D} = x

@inline function (f::InitMin{D})(x, y) where {D}
    min(x, y, D)
end

@inline function (f::InitMax{D})(x, y) where {D}
    max(x, y, D)
end

Base.isinf(x::Finch.Square) = isinf(x.arg) || isinf(x.scale)
Base.isinf(x::Finch.Power) = isinf(x.arg) || isinf(x.scale) || isinf(x.exponent)

"""
initmax(z)(a, b)
initmin(z)(a, b)

`initmax(z)` is a function which may assert that `a`
[`isequal`](https://docs.julialang.org/en/v1/base/base/#Base.isequal) to `z`,
and `returns `b`.  By default, `lhs[] = rhs` is equivalent to `lhs[]
<<initwrite(default(lhs))>>= rhs`.
"""
initmax(z) = InitMax{z}()
initmin(z) = InitMin{z}()

Finch.isidentity(::Finch.AbstractAlgebra, ::InitMax{D}, x) where {D} = isequal(x, D)
Finch.isidentity(::Finch.AbstractAlgebra, ::InitMin{D}, x) where {D} = isequal(x, D)
