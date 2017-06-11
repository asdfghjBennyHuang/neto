# neto

Neto is a module that provide helper functions to store Artificial Neural Networks (ANN) and compiled function objects built with lasagne and theano.

### functions

`store_compiled_fn(fns, filename, old_dir='old_file') -> None`

store the functions object compiled with `theano.function`. 

fns : list or function object

filename : str<br>
    the filename(or path) to store the functions, since it uses pickle to store the objects, suggested file extension is '.pic'.

old_dir : str<br>
    if the filename exist, move the existing file into this directory, in order to preserve previously compiled functions
  
`load_compiled_fn(filename) -> tuple`

return the tuple that contains the functions stored in the file
  
fileanme : str<br>
  
`store_network(net, filename, old_dir='old_file') -> None`

store the parameters of network into the file, note that next time you still have to build the network with lasagne, than you can use `load_network` function to load the parameter into the network

net : lasagne layer object

filename : str<br>
    the filename(or path) to store the parameters.

old_dir : str<br>
    if the filename exist, move the existing file into this directory, in order to preserve previously compiled functions

`load_network(net, filename) -> None`

load the network with the parameters stored in file
 
net : lasagne layer object

filename : str<br>
    the filename(or path) to store the parameters.
