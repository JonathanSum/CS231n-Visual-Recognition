import math

x = 3  # example values
y = -4

# forward pass
sigy = 1.0 / (1 + math.exp(-y))  # sigmoid in numerator   #(1)
num = x + sigy  # numerator                               #(2)
sigx = 1.0 / (1 + math.exp(-x))  # sigmoid in denominator #(3)
xpy = x + y  # (4)
xpysqr = xpy ** 2  # (5)
den = sigx + xpysqr  # denominator                        #(6)
invden = 1.0 / den  # (7)
f = num * invden  # done!
# I am a line------------------------------------------------
# f(x,y)=(x+sigy)*invden or num*invden



# backdroup round 1: f=m1*m2 m1=num, m2=invden
dnum = invden
dinvden = num

# backdroup about the dinvden round2 2: f(x)=1/x
dden = dinvden * (-1.0 / den ** 2)

# round3: two parts
dxpysqr = 1 * dden
dsigx = 1 * dden
# round4: dxpy**2 and the sigx
dxpy = 2 * (xpy) * dxpysqr
dx = dxpy
dy = dxpy
dx += (1 - sigx) * sigx * dsigx
dsigy = dnum
dy += (1 - sigy) * sigy * dsigy
dx += dnum

print("{0} {1}".format(dx, dy))
