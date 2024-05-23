# This file manages the logic of algebraic properties. For the moment, we just spoof Finch's
# properties, but in the future we can extend these or write our own.

function isassociative(f)
    return Finch.isassociative(Finch.DefaultAlgebra(), f)
end
function iscommutative(f)
     return Finch.iscommutative(Finch.DefaultAlgebra(), f)
end
function isdistributive(f, g)
#    distributes = Finch.isdistributive(Finch.DefaultAlgebra(), f, g)
    distributes = (((g == max) || (g == min)) && (f == +))
    distributes = distributes || ((f == &) && (g == |))
    distributes = distributes || ((f == *) && (g == +))
    return distributes
end

function isidentity(f, x)
    return Finch.isidentity(Finch.DefaultAlgebra(), f, x)
end

function isannihilator(f, x)
    return Finch.isannihilator(Finch.DefaultAlgebra(), f, x)
end

function isidempotent(f)
    return Finch.isidempotent(Finch.DefaultAlgebra(), f)
end
