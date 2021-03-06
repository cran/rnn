## ----int2bin------------------------------------------------------------------
# basic conversion
i2b <- function(integer, length=8)
  as.numeric(intToBits(integer))[1:length]

# apply to entire vectors
int2bin <- function(integer, length=8)
  t(sapply(integer, i2b, length=length))

## ----data---------------------------------------------------------------------
# set training data length
training_data_size = 20000

# create sample inputs
X1 = sample(0:127, training_data_size, replace=TRUE)
X2 = sample(0:127, training_data_size, replace=TRUE)

# create sample output
Y <- X1 + X2

# convert to binary
X1 <- int2bin(X1)
X2 <- int2bin(X2)
Y  <- int2bin(Y)

# create 3d array: dim 1: samples; dim 2: time; dim 3: variables
X <- array( c(X1,X2), dim=c(dim(X1),2) )

## ----sigmoid------------------------------------------------------------------
sigmoid <- function(x)
             1 / ( 1+exp(-x) )


sig_to_der <- function(x)
                x*(1-x)

## ----hyperparameters----------------------------------------------------------
binary_dim = 8
alpha      = 0.5
input_dim  = 2
hidden_dim = 6
output_dim = 1

## ----weights-init-------------------------------------------------------------

# initialize weights randomly between -1 and 1, with mean 0
weights_0 = matrix(runif(n = input_dim *hidden_dim, min=-1, max=1),
                   nrow=input_dim,
                   ncol=hidden_dim ) 
weights_h = matrix(runif(n = hidden_dim*hidden_dim, min=-1, max=1),
                   nrow=hidden_dim,
                   ncol=hidden_dim )
weights_1 = matrix(runif(n = hidden_dim*output_dim, min=-1, max=1),
                   nrow=hidden_dim,
                   ncol=output_dim ) 

# create matrices to store updates, to be used in backpropagation
weights_0_update = matrix(0, nrow = input_dim,  ncol = hidden_dim) 
weights_h_update = matrix(0, nrow = hidden_dim, ncol = hidden_dim)
weights_1_update = matrix(0, nrow = hidden_dim, ncol = output_dim)

## ----training-----------------------------------------------------------------

# training logic
for (j in 1:training_data_size) {
    # select data
    a = X1[j,]
    b = X2[j,]
    
    # select true answer
    c = Y[j,]
    
    # where we'll store our best guesss (binary encoded)
    d = matrix(0, nrow = 1, ncol = binary_dim)
    
    overallError = 0
    
    layer_2_deltas = matrix(0)
    layer_1_values = matrix(0, nrow=1, ncol = hidden_dim)

    # moving along the positions in the binary encoding
    for (position in 1:binary_dim) {
        # generate input and output
        X = cbind( a[position], b[position] ) # rename X to layer_0?
        y = c[position]

        # hidden layer
        layer_1 = sigmoid( (X%*%weights_0) +
                    (layer_1_values[dim(layer_1_values)[1],] %*% weights_h) )
    
        # output layer
        layer_2 = sigmoid(layer_1 %*% weights_1)
    
        # did we miss?... if so, by how much?
        layer_2_error = y - layer_2
        layer_2_deltas = rbind(layer_2_deltas, layer_2_error * sig_to_der(layer_2))
        overallError = overallError + abs(layer_2_error)
    
        # decode estimate so we can print it out
        d[position] = round(layer_2)
        
        # store hidden layer
        layer_1_values = rbind(layer_1_values, layer_1)
    }

    future_layer_1_delta = matrix(0, nrow = 1, ncol = hidden_dim)
    
    for (position in binary_dim:1) {
        X = cbind(a[position], b[position])
        layer_1 = layer_1_values[dim(layer_1_values)[1]-(binary_dim-position),]
        prev_layer_1 = layer_1_values[dim(layer_1_values)[1]- ( (binary_dim-position)+1 ),]
        
        # error at output layer
        layer_2_delta = layer_2_deltas[dim(layer_2_deltas)[1]-(binary_dim-position),]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta %*% t(weights_h) +
          layer_2_delta %*% t(weights_1)) * sig_to_der(layer_1)
    
        # let's update all our weights so we can try again
        weights_1_update = weights_1_update + matrix(layer_1) %*% layer_2_delta
        weights_h_update = weights_h_update + matrix(prev_layer_1) %*% layer_1_delta
        weights_0_update = weights_0_update + t(X) %*% layer_1_delta
    
        future_layer_1_delta = layer_1_delta
    }
    
    weights_0 = weights_0 + ( weights_0_update * alpha )
    weights_1 = weights_1 + ( weights_1_update * alpha )
    weights_h = weights_h + ( weights_h_update * alpha )
    
    weights_0_update = weights_0_update * 0
    weights_1_update = weights_1_update * 0
    weights_h_update = weights_h_update * 0
    
    if(j%%(training_data_size/10) == 0)
        print(paste("Error:", overallError))    

}

