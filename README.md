This repository consists of many matrix multiplication algorithms.

## History ##
2022-08-01 Strassen's algorithm added
2022-07-25 Git repository uploaded to the Git

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

|Test case| Algorithm 1 | Algorithm 2 | Algorithm 3 | Algorithm 4 | Algorithm 5 | Algorithm 6 | Algorithm 7 | Algorithm 8 | Algorithm 9 | Algorithm 10 |
| --- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | 
| (128 x 128) x <br>(128 x 128) =><br> (128 x 128) | 7.72624 | 1.52 | 1.49 | 1.78 | 1.67 | 1.60 | 1.57 | 1.58 | 1.56 | 1.50 |
| (1024 x 1024) x <br>(1024 x 1024) =><br> (1024 x 1024) | 4,978.36 | 26.07 | 17.88 | 68.35 | 67.70 | 35.16 | 32.35 | 21.92 | 19.57 | 18.20 |
| (4096 x 4096) x <br>(4096 x 4096) =><br> (4096 x 4096) | - | 1208.86 | 919.98 | 2533.22 | 2533.13 | 1058.58 | 952.16 | 488.61 | 379.03 | 398.82 |
| (16000 x 16000) x <br>(16000 x 16000) =><br> (16000 x 16000) | - | 86023.20 | 71400.70 | 151551.00 | 151413.00 | 59845.10 | 52840.80 | 24951.50 | 18331.90 | 11931.20 |
| (4096 x 10) x <br>(10 x 4096) =><br> (4096 x 4096) | - | 61.89 | 59.82 | 61.20 | 61.35 | 63.69 | 63.83 | 64.54 | 63.16 | 60.96 | 
| (16000 x 100) x <br>(100 x 16000) =><br> (16000 x 16000) | - |  911.15 | 787.60 | 1016.16 | 596.63 | 951.94 | 909.46 | 737.04 | 696.80 | 863.41 |
| (100 x 16000) x <br>(16000 x 16000) =><br> (100 x 16000) | - | 745.62 | 643.88 | 1190.49 | 141361 | 691.65 | 645.89 | 417.54 | 356.97 | 636.31 |
| (16000 x 16000) x <br>(16000 x 100) =><br> (16000 x 100) | - | 577.03 | 417.98 | 1265.97 | 1558.14 | 768.10 | 738.41 | 514.09 | 452.06 | 536.16 |
| (100 x 100) x <br>(100 x 32000) =><br> (100 x 32000) | - | 15.11 | 14.29 | 17.84 | 10.00 | 17.61 | 16.86 | 13.76 | 12.48 | 14.29 |
| (100 x 32000) x <br>(32000 x 100) =><br> (100 x 100) | - | 110.39 | 99.94 | 125.76 | 149.63 | 113.57 | 112.62 | 103.34 | 102.57 | 99.87 |
| (32000 x 100) x <br>(100 x 100) =><br> (32000 x 100) | - | 243.78 | 247.62 | 254.87 | 242.43 | 250.83 | 248.60 | 243.25 | 244.34 | 244.23 |

![Experiment result](https://github.com/aintmos/Matrix-multiplications/blob/main/doc/ExperimentResult.png?raw=true)
