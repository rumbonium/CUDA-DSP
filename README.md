# Overview
These programs implement a software defined reciever that isolates one FM station using multiple stages of decimating FIR filters.  One FM station is demodulated using the quadrature demodulation technique.  There are three programs.
1. The first program uses the UHD to control the USRP hardware and writes to shared memory.
2. The second program reads from shared memory, process the signal on the GPU using the C-API for CUDA, and writes the output to standard out.
3. The third program (SoX) receives the signal through a pipe and reads from standard input.  SoX writes the audio to the sound card.

# Computer hardware information
We are using Ubuntu 16.04 LTS and a Nvidia GeForce GTX 1080 Ti GPU.

# Radio hardware information
We are using a generic scanner antenna (25-1300 MHz) connected to a custom bandpass filter (KR Electronics) passing the 20 MHz wide FM band (88-108 MHz).  The BPF output is connected to a USRP B205mini-i, which is connected to the PC via USB3.0.  The programs below shift the center of the FM band to zero frequency (zero-IF receiver) and one station is isolated using multirate dignal signal processing implemented on the GPU.  An FM station is demodulated and sample rate converted to 40 kSamples/second.  The audio signal is piped to Sox for listening.

# Software, drivers, libraries
These programs require the following software:
 - [USRP Hardware Driver (UHD)](https://files.ettus.com/manual/index.html)
 - Boost C++ library (should be pre-installed)
 - [CUDA (we use CUDA 8.0)](https://developer.nvidia.com/cuda-zone)
 - [SoX - Sound eXchange](http://sox.sourceforge.net/)

# DSP program:
Compile Instructions:
```linux
nvcc dsp4.cu -o dsp4
```  

Execute Instructions: *(NOTE: Launch this program before `usrp_stream4`.  The output of this program is piped to SoX for audio.)*
```linux
./dsp4 | play --rate 40k -b 32 -c 1 -e float -t raw -
```

# Usrp stream program:
Compile Instructions:
```linux
usrp_stream4.cpp -o usrp_stream4 -luhd -lboost_system -lboost_thread -lboost_program_options
```

Execute Instructions: *(NOTE: Launch this program after the `dsp4`.)*
```linux
./usrp_stream4.cpp --rate 20000000 --freq 98000000 --spb 16384 --type float
```
