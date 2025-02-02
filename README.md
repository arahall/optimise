this is a pure C++ implementation of the LBFGSB algorithm. The code uses [EIGEN](https://eigen.tuxfamily.org/index.php?title=Main_Page) for matrix and vector
operations but if this is not available then a home-rolled matrix library is used. To use EIGEN specify
the USING_EIGEN in the pre-processor options. 

It has been tested with a wide range of standard optimser tests e.g. rosenbrock, matyas, rastrigin, beale etc

Separately, the code can be called from python using the ctypes library. this requires a DLL wrapper in windows 
to marshall the data and callback between python and the optimisation module. It has been compared to the scipy minimise 
implementation of LBFGSB and for most large problems it outperforms scipy sometimes by a factor of 2. 

i've also used the code to calibrate a Hawkes process (3 paramters, 500 observattions) by minimising the log likelihood function and compared
the run time with scipy. for this problem scipy seems to be about 40% faster.
