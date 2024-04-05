# This file manages the logic of algebraic properties. For the moment, we just spoof Finch's
# properties, but in the future we can extend these or write our own.

function isassociative(f)
    return Finch.isassociative(Finch.DefaultAlgebra(), f)
end
function iscommutative(f)
     return Finch.iscommutative(Finch.DefaultAlgebra(), f)
end
function isdistributive(f, g)
    return Finch.isdistributive(Finch.DefaultAlgebra(), f, g)
end
