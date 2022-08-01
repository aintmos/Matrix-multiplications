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
5. X,Y-coordinated GPU implementated blocked dense-matrix multiplication + Registered result
6. X,Y-coordinated GPU implementated blocked dense-matrix multiplication + Registered result + Utilize shared memory
7. X,Y-coordinated GPU implementated blocked dense-matrix multiplication + Registered result + Utilize shared memory + Shared memory indexing method 1
8. X,Y-coordinated GPU implementated blocked dense-matrix multiplication + Registered result + Utilize shared memory + Shared memory indexing method 2
9. X,Y-coordinated GPU implementated blocked dense-matrix multiplication + Registered result + Utilize shared memory + Shared memory indexing method 3
10. Strassen's algorithm
11. (WIP) Winograd's algorithm
12. (WIP) Cublas

#### Experiment setup ####

| GPU | CPU | RAM | OS |
| --- | --- | --- | --- |
| GeForce RTX 2060 | Intel(R) Core(TM) i7-6700 | 16.341768 GB | Ubuntu |

#### Experiment result ####

| Index |Test case| Algorithm 1(msecs) | Algorithm 2(msecs) | Algorithm 3(msecs) | Algorithm 4(msecs) | Algorithm 5(msecs) | 
| --- | --- | ----------- | ----------- | ----------- | ----------- | ----------- |
| 1 | (128 x 128) x <br>(128 x 128) =><br> (128 x 128) | 7.72624 | 1.52 | 1.49 | 1.78 | 1.67 |
| 2 | (1024 x 1024) x <br>(1024 x 1024) =><br> (1024 x 1024) | 4,978.36 | 26.07 | 17.88 | 68.35 | 67.70 | 
| 3 | (4096 x 4096) x <br>(4096 x 4096) =><br> (4096 x 4096) | (Predicted) 253K | 1208.86 | 919.98 | 2533.22 | 2533.13 |
| 4 | (16000 x 16000) x <br>(16000 x 16000) =><br> (16000 x 16000) | (Predicted) 15M | 86K| 71.4K | 151.55K | 151.41K | 
| 5 | (4096 x 10) x <br>(10 x 4096) =><br> (4096 x 4096) | 618 | (Predicted) 61.89 | 59.82 | 61.20 | 61.35 | 
| 6 | (16000 x 100) x <br>(100 x 16000) =><br> (16000 x 16000) | (Predicted) 94K |  911.15 | 787.60 | 1016.16 | 596.63 | 
| 7 | (100 x 16000) x <br>(16000 x 16000) =><br> (100 x 16000) | (Predicted) 94K | 745.62 | 643.88 | 1190.49 | 1413.61 |
| 8 | (16000 x 16000) x <br>(16000 x 100) =><br> (16000 x 100) | (Predicted) 94K | 577.03 | 417.98 | 1265.97 | 1558.14 |
| 9 | (100 x 100) x <br>(100 x 32000) =><br> (100 x 32000) | (Predicted) 1178 | 15.11 | 14.29 | 17.84 | 10.00 |
| 10 | (100 x 32000) x <br>(32000 x 100) =><br> (100 x 100) | (Predicted) 1178 | 110.39 | 99.94 | 125.76 | 149.63 |
| 11 | (32000 x 100) x <br>(100 x 100) =><br> (32000 x 100) | (Predicted) 1178 | 243.78 | 247.62 | 254.87 | 242.43 |


| Index | Test case | Algorithm 6(msecs) | Algorithm 7(msecs) | Algorithm 8(msecs) | Algorithm 9(msecs) | Algorithm 10(msecs) |
| --- | --- | ----------- | ----------- | ----------- | ----------- | ----------- | 
| 1 | (128 x 128) x <br>(128 x 128) =><br> (128 x 128) | 1.60 | 1.57 | 1.58 | 1.56 | 1.50 |
| 2 | (1024 x 1024) x <br>(1024 x 1024) =><br> (1024 x 1024) | 35.16 | 32.35 | 21.92 | 19.57 | 18.20 |
| 3 | (4096 x 4096) x <br>(4096 x 4096) =><br> (4096 x 4096) | 1058.58 | 952.16 | 488.61 | 379.03 | 398.82 |
| 4 | (16000 x 16000) x <br>(16000 x 16000) =><br> (16000 x 16000) | 59.84K | 52.840K | 24.951K | 18.331K | 11.931K |
| 5 | (4096 x 10) x <br>(10 x 4096) =><br> (4096 x 4096) | 63.69 | 63.83 | 64.54 | 63.16 | 60.96 | 
| 6 | (16000 x 100) x <br>(100 x 16000) =><br> (16000 x 16000) | 951.94 | 909.46 | 737.04 | 696.80 | 863.41 |
| 7 | (100 x 16000) x <br>(16000 x 16000) =><br> (100 x 16000) | 691.65 | 645.89 | 417.54 | 356.97 | 636.31 |
| 8 | (16000 x 16000) x <br>(16000 x 100) =><br> (16000 x 100) | 768.10 | 738.41 | 514.09 | 452.06 | 536.16 |
| 9 | (100 x 100) x <br>(100 x 32000) =><br> (100 x 32000) | 17.61 | 16.86 | 13.76 | 12.48 | 14.29 |
| 10 | (100 x 32000) x <br>(32000 x 100) =><br> (100 x 100) | 113.57 | 112.62 | 103.34 | 102.57 | 99.87 |
| 11 | (32000 x 100) x <br>(100 x 100) =><br> (32000 x 100) | 250.83 | 248.60 | 243.25 | 244.34 | 244.23 |

#### Ranking ####
![Experiment result](https://github.com/aintmos/Matrix-multiplications/blob/main/doc/ExperimentResult.png?raw=true)
