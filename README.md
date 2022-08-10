This repository consists of many matrix multiplication algorithms.

## History ##
- 2022-08-01 Strassen's algorithm added
- 2022-07-25 Git repository uploaded to the Git

## Algorithms ##

### DMM ###
1. Naive CPU implemented dense-matrix multiplication
2. Naive GPU implemented dense-matrix multiplication
3. Naive GPU implemented dense-matrix multiplication + Registered result 
4. X,Y-coordinated GPU implementated dense-matrix multiplication + Registered result
5. X,Y-coordinated GPU implementated blocked dense-matrix multiplication + Registered result + Utilize shared memory
6. X,Y-coordinated GPU implementated blocked dense-matrix multiplication + Registered result + Utilize shared memory + Shared memory indexing method 1
7. X,Y-coordinated GPU implementated blocked dense-matrix multiplication + Registered result + Utilize shared memory + Shared memory indexing method 2
8. X,Y-coordinated GPU implementated blocked dense-matrix multiplication + Registered result + Utilize shared memory + Shared memory indexing method 3
9. Strassen's algorithm
10. Memory usage optimized Strassen's algorithm
11. Winograd's algorithm
12. Blocked algorithm
13. (WIP) Cublas

#### Experiment setup ####

| GPU | CPU | RAM | OS |
| --- | --- | --- | --- |
| GeForce RTX 2060 | Intel(R) Core(TM) i7-6700 | 16.341768 GB | Ubuntu |

#### Experiment result ####

##### Test case #####
| Index | Matrix size | Input size | Result size | Remark |
| ----- | ----------- | ---------- | ----------- | ------ |
| 1 | 128 x 128 | 128 x 128 | 128 x 128 | Rectangular matrix/input 1 |
| 2 | 1024 x 1024 | 1024 x 1024 | 1024 x 1024 | Rectangular matrix/input 1 |
| 3 | 4096 x 4096 | 4096 x 4096 | 4096 x 4096 | Rectangular matrix/input 2 |
| 4 | 16000 x 16000 | 16000 x 16000 | 16000 x 16000 | Rectangular matrix/input 3 |
| 5 | 16383 x 16383 | 16383 x 16383 | 16383 x 16383 | Rectangular matrix/input 4 (Allways odd size with recursion) |
| 6 | 4096 x 10 | 10 x 4096 | 4096 x 4096 | Small square matrix/input |
| 7 | 100 x 16000 | 16000 x 16000 | 100 x 16000 | Square matrix|
| 8 | 16000 x 100 | 100 x 16000 | 16000 x 16000 | Square matrix/input |
| 9 | 16000 x 16000 | 16000 x 100 | 16000 x 100 | Square input |
| 10 | 100 x 100 | 100 x 32000 | 100 x 32000 | Form 1 |
| 11 | 100 x 32000 | 32000 x 100 | 100 x 100 | Form 2 |
| 12 | 32000 x 100 | 100 x 100 | 32000 x 100 | Form 3 |

##### Test results #####
| Index | MM1(msec) | MM2(msec) | MM3(msec) | MM4(msec) | MM5(msec) | MM6(msec) |
| ----- | --------- | --------- | --------- | --------- | --------- | --------- |
| 0 | 7.7 | 1.61 | 1.66 | 1.71 | 1.73 | 1.72 | 
| 1 | 9.07K | 26.27 | 18.27 | 67.4 | 74.37 | 68.98 | 
| 2 | 252.31K | 937.43 | 804.42 | 2.55K | 2.83K | 2.76K | 
| 3 | 15.04M | 85.92K | 71.55K | 154.53K | 165.65K | 161.37K | 
| 4 | 16.14M | 97.72K | 98.13K | 85.23K | 171.26K | 166.65K | 
| 5 | 615.99 | 60.36 | 58.82 | 59.91 | 67.84 | 67.78 | 
| 6 | 93.99K | 741.45 | 644.77 | 1.2K | 1.55K | 1.5K | 
| 7 | 93.99K | 877.57 | 782.96 | 1.02K | 1.64K | 1.62K | 
| 8 | 93.99K | 574.45 | 428.97 | 1.3K | 1.64K | 1.61K | 
| 9 | 1.17K | 14.69 | 14.18 | 15.37 | 26.98 | 26.44 | 
| 10 | 1.17K | 108.27 | 100.6 | 15.37 | 136.01 | 141.98 | 
| 11 | 1.17K | 245.13 | 243.7 | 249.53 | 265.61 | 264.71 |

| Index | MM7(msec) | MM8(msec) | MM9(msec) | MM10(msec) | MM11(msec) | MM12(msec) |
| ----- | --------- | --------- | --------- | ---------- | ---------- |---------- | 
| 0 | 1.55 | 1.54 | 1.52 | 1.52 | 1.61 | 1.60 |
| 1 | 22.15 | 20.48 | 18.11 | 18.06 | 18.21 | 18.14 |
| 2 | 492.91 | 422.09 | 320.62 | 315.96 | 305.43 | 345.61 |
| 3 | 24.83K | 20.59K | 12.04K | 11.92K | 11.38K | 1.69K |
| 4 | 21.0K | 16.14K | 13.87K | 13.57K | 12.89K | 18.83K |
| 5 | 59.54 | 60.92 | 57.83 | 58.55 | 60.08 | 59.77 |
| 6 | 430.0 | 392.0 | 633.08 | 621.75 | 596.2 | 596.23 |
| 7 | 768.82 | 739.29 | 781.01 | 780.77 | 772.6 | 765.33 |
| 8 | 513.0 | 484.94 | 522.84 | 509.67 | 478.36 | 454.79 |
| 9 | 12.23 | 11.78 | 14.11 | 14.21 | 14.2 | 14.16 |
| 10 | 103.53 | 102.3 | 101.97 | 102.28 | 102.05 | 102.73 |
| 11 | 244.63 | 243.32 | 244.74 | 245.57 | 245.21 | 243.28 |

#### Charts ####
![Experiment result](https://github.com/aintmos/Matrix-multiplications/blob/main/doc/Result.png?raw=true)

![Experiment result](https://github.com/aintmos/Matrix-multiplications/blob/main/doc/Result_drop_some.png?raw=true)

![Experiment result](https://github.com/aintmos/Matrix-multiplications/blob/main/doc/Throughput.png?raw=true)

![Experiment result](https://github.com/aintmos/Matrix-multiplications/blob/main/doc/Throughput_drop_some.png?raw=true)

![Experiment result](https://github.com/aintmos/Matrix-multiplications/blob/main/doc/Testcase5.png?raw=true)

![Experiment result](https://github.com/aintmos/Matrix-multiplications/blob/main/doc/Testcase5_drop_some.png?raw=true)

![Experiment result](https://github.com/aintmos/Matrix-multiplications/blob/main/doc/Rank.png?raw=true)
