## ----package-------------------------------------------------------------
library(rnn)

## ----code-rnn------------------------------------------------------------
rnn

## ----int2binary----------------------------------------------------------
int2binary(146, length=8)

## ----int2binary-code-----------------------------------------------------
int2binary

## ----sigmoid-------------------------------------------------------------
(a <- sigmoid(3))

## ----sigmoid-code--------------------------------------------------------
sigmoid

## ----sigmoid-der---------------------------------------------------------
sigmoid_output_to_derivative(a) # a was created above using sigmoid()

## ----sigmoid-der-code----------------------------------------------------
sigmoid_output_to_derivative

## ----seed----------------------------------------------------------------
set.seed(123)

## ----help, eval=FALSE----------------------------------------------------
#  help('rnn')

## ----data----------------------------------------------------------------
# create sample inputs
X1 = sample(0:127, 5000, replace=TRUE)
X2 = sample(0:127, 5000, replace=TRUE)

# create sample output
Y <- X1 + X2

## ----example-------------------------------------------------------------
rnn(Y,
    X1,
    X2,
    binary_dim =  8,
    alpha      =  0.1,
    input_dim  =  2,
    hidden_dim = 10,
    output_dim =  1)

## ----example-2-----------------------------------------------------------
# create sample inputs
X1 = sample(0:127, 20000, replace=TRUE)
X2 = sample(0:127, 20000, replace=TRUE)

# create sample output
Y <- X1 + X2

rnn(Y,
    X1,
    X2,
    binary_dim = 8,
    alpha      = 0.1,
    input_dim  = 2,
    hidden_dim = 3,
    output_dim = 1)

