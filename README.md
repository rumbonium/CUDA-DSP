# CUDA-DSP
Cuda dsp program:
  Included libraries (the libraries that need to be installed to run this program):
  <fcntl.h>
  <sys/stat.h>
  <sys/types.h>
  <unistd.h>
  <stdio.h>
  <cuda_runtime.h>
  <stdlib.h>
  <math.h>
  <string.h>
  <sys/types.h>
  <sys/ipc.h>
  <sys/shm.h>
  <sys/sem.h>
  <unistd.h>
  <complex>
  Also install sox
  
  Included files (keep these files in the same directory as dsp4.cu):
  "filt1.h"
  "filt2.h"
  "filt3.h"
  "filt4.h"
  "filtd.h"
  "filt7.h"
  
  Compile Instructions:
  Compile command: "nvcc dsp4.cu -o dsp4"
  
  Execute Instructions: Run this program first. The 'stdout' of this program is piped to sox for audio.
  Execute command: "./dsp4 | play --rate 40k -b 32 -c 1 -e float -t raw -"


Usrp stream program:
  Included libraries (the libraries that need to be installed to run this program):
  <uhd/types/tune_request.hpp>
  <uhd/utils/thread_priority.hpp>
  <uhd/utils/safe_main.hpp>
  <uhd/usrp/multi_usrp.hpp>
  <uhd/exception.hpp>
  <boost/program_options.hpp>
  <boost/format.hpp>
  <boost/thread.hpp>
  <iostream>
  <fstream>
  <csignal>
  <complex>
  <fcntl.h>
  <sys/stat.h>
  <sys/types.h>
  <unistd.h>
  <algorithm>
  <sys/types.h>
  <sys/ipc.h>
  <sys/shm.h>
  <sys/sem.h>
  <stdio.h>
  <stdlib.h>
  <unistd.h>
  <string.h>
  <stdlib.h>

  Compile Instructions:
  Compile command: "usrp_stream4.cpp -o usrp_stream4 -luhd -lboost_system -lboost_thread -lboost_program_options"
  
  Execute Instructions: Run this program second
  Execute command: "./usrp_stream4.cpp --rate 20000000 --freq 98000000 --spb 16384 --type float"
