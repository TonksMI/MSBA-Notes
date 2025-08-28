# Calculus and Linear algebra

## Derivatives 

Instantaneos rate of change

slope of a line at a point

we care when derivatives are zero

relative mins/maxs

we want to maximize/minimize models of data

## Integrals

Areas under the curve

Probablily curves use integrals to find probability of a given Z score

- the area under a curve will always be zero

Area under a curve

- Gradients

derivitaves of multivariable function

partial derivatives

shove the different functions into a vector and you make a gradient

## Important Matrices

Covariance
Correlation
Hessian
Design Matrix

``` r
corr(x,y) =
Covariance x,y
sd x * sd y
```

### Hessian

gradients: first derivatives of a multivariable function 𝑓(𝑥,𝑦)

hessians : second derivatives of a multivariable function 𝑓(𝑥,𝑦)

Position: 𝑟(𝑡)

Velocity: 𝑑𝑟𝑑𝑡

Acceleration: 𝑑2𝑟𝑑𝑡2

Hessian
𝑓=𝑥2+𝑥𝑦+𝑦2

∂𝑓∂𝑥=2𝑥+𝑦;∂𝑓∂𝑦=𝑥+2𝑦
⎡⎣⎢⎢∂2𝑓∂𝑥2∂𝑓2∂𝑦∂𝑥∂2𝑓∂𝑥∂𝑦∂2𝑓∂𝑦2⎤⎦⎥⎥=[2112]

Second derivative is the acceleration of a point of velocity

## Design Matrix

is a linear model is a matrix of all explanatory variables

y = Xb

simple case: columns of X are vextors of predictor vairables

complicated case: on-hot encoding, intercept, polynomial terms

using a desing matrix X makes writing out the math easier. Y is usually a 1xN vector of responses 

## Least Squares

y = XB

oftern no B can perfect prediction of y or if a model does it probably is overfitting

what are all the possible linear regression that are possible. 

we choose a line in the column space of X that is as close as possible

THe best we can do is both in the column space of X
and minimizes the distance between Y and XB

is the projection of Y on C(X)

So our best fit is the XB= projc(x)y

if we take y from both size

colinearity two predictors are sort of connected or dependent on each other

LOOK UP XGBOOST

## Determinants

Transformation of that matrix square after multiplying the

## Important Matrix Decompositions

Singular value
Eigenvalue
QR
Cholesky

### Singular value Decomposition

breaks down a matrix of U

DIagonal matrix are sparse. Most entries are zero. Computationaly easy

𝑈𝑈 an orthogonal matrix; eigenvectors of 𝐴𝐴𝑇𝐴𝐴𝑇
𝐷𝐷 is a diagonal matrix; diagonal is root of positive eigenvalues of 𝐴𝐴𝑇
𝐴𝐴𝑇 or 𝐴𝑇𝐴𝐴𝑇𝐴
𝑉𝑉 an orthogonal matrix; eigenvectors of 𝐴𝑇𝐴𝐴𝑇𝐴

Decomposing the data matrix XX rather than eigen decomposing the covariance matrix

eigen vectors 

### Eigendecomposition

P - gives a matrix of eigenvectors
N - diagonal matrix of eigenvalues

A = P * N * P-1

### QR Decomposition

Q orthogonal matrix
RR upper triangular matrix

A = Q*R

### Cholesky Decomposition

Square root of matrix

LL lower triangular matrix
LtLt is an upper triangular matrix


A = LL * L

## Taylor Series Expansion

match derivative of a function iteratively to approximate a function

also match important characteristics

