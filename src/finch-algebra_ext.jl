# This file declares some important algebraic properties of major functions
# so that Finch can produce efficient code.
Finch.iscommutative(::Finch.DefaultAlgebra, ::typeof(max)) = true
Finch.isassociative(::Finch.DefaultAlgebra, ::typeof(max)) = true
Finch.isinvolution(::Finch.DefaultAlgebra, ::typeof(max)) = true
Finch.isidempotent(::Finch.DefaultAlgebra, ::typeof(max)) = true
Finch.isdistributive(::Finch.DefaultAlgebra, ::typeof(+), ::typeof(max))= true

Finch.iscommutative(::Finch.DefaultAlgebra, ::typeof(min)) = true
Finch.isassociative(::Finch.DefaultAlgebra, ::typeof(min)) = true
Finch.isinvolution(::Finch.DefaultAlgebra, ::typeof(min)) = true
Finch.isidempotent(::Finch.DefaultAlgebra, ::typeof(min)) = true
Finch.isdistributive(::Finch.DefaultAlgebra, ::typeof(+), ::typeof(min))= true
