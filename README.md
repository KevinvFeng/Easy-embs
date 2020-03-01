# Easy-embs
This is a C++ implementation of word2vec and relative models. It takes parallel technique to speedup training. 

The code is developed based on the [original word2vec](https://code.google.com/archive/p/word2vec/)  implementation from Google and [pWord2Vec](https://github.com/IntelLabs/pWord2Vec) implementation from intel.

## License
All source code files in the package are under [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).

## Prerequisites
The code is developed and tested on UNIX-based systems with the following software dependencies:

- [Intel Compiler](https://software.intel.com/en-us/qualify-for-free-software) (The code is optimized on Intel CPUs)
- OpenMP (No separated installation is needed once Intel compiler is installed)
- MPI library, with multi-threading support (Intel MPI, MPICH2 or MVAPICH2 for distributed word2vec only)
- [HyperWords](https://bitbucket.org/omerlevy/hyperwords) (for model accuracy evaluation)
- Numactl package (for multi-socket NUMA systems)

## Environment Setup
* Install Intel C++ development environment (i.e., Intel compiler, OpenMP, MKL "16.0.0 or higher" and iMPI. [free copies](https://software.intel.com/en-us/qualify-for-free-software) are available for some users)
* Enable Intel C++ development environment
```
source /opt/intel/bin/iccvars.sh intel64
source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64 (please point to the path of your installation)
source /opt/intel/impi/latest/compilers_and_libraries/linux/bin/compilervars.sh intel64 (please point to the path of your installation)
```
* Install numactl package
```
sudo yum install numactl (on RedHat/Centos)
sudo apt-get install numactl (on Ubuntu)
```

## Quick Start
1. Download the code: ```git clone https://github.com/KevinvFeng/Easy-embs_based_on_pWord2Vec.git```
2. Run .\install.sh to build the package (e.g., it downloads hyperwords and compiles the source code.)  
Note that this installation will try to produce two binaries: pWord2Vec and pWord2Vec_mpi. If you are only interested in the non-mpi version of w2v, you don't need to set up mpi and the compilation will fail on building pWord2Vec_mpi of course. But you can still use the non-mpi binary for the rest of single machine demos.
3. Run shell command ```make clean all```
4. Run command ```numactl ./pWord2Vec -train ./path/to/dataset -output vectors.txt -size 100 -window 8 -negative 5 -sample 1e-4 -binary 0 -iter 8 -min-count 5 -save-vocab vocab.txt -batch-size 17 -cwe-type 0 -threads 8 -cbow 1```

## Future Works
- Support MKL 
