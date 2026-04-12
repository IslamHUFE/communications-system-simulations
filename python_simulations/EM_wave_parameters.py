from mpmath import inf, mp

mp.dps = 50


class constants:
    ε0 = 8.8541878128e-12
    µ0 = 1.25663706212e-6
    c = 299792458


print("What it the type of the medium?")
print("1. Free space (σ = 0 , ε=εo , µ=µo)")
print("2. Lossless dielectrics (σ ≃ 0 , ε= ε r εo , µ= µr µ o, or σ<<ωε)")
print("3. Lossy dielectrics (σ ≠ 0 , ε= εr εo, µ= µr µo)")
print("4. Good conductors (σ ≃ ∞ , ε = εo, µ= µr µ o , or σ >> ωε)")

medium = input(">>")

if medium == "1":
    σ = 0
    ε = constants.ε0
    µ = constants.µ0
    print("what is missing?")
    print("1. α | 2. β | 3. η | 4. θη | 5. ν | 6. ω")
elif medium == "2":
    σ = 0
    print("what is missing?")
elif medium == "3":
    print("what is missing?")
elif medium == "4":
    σ = inf
    print("what is missing?")
else:
    print("Invalid input")
