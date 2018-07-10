from pyculib import fft
import numpy as np
import time
from numba import cuda
# img_shape = (5760, 4092)
img_shape = (4096, 4096)
img = np.random.standard_normal(img_shape).astype(np.complex64)
img_out = img.astype(np.complex64)

#f = fft.FFTPlan(shape=img_shape, itype=np.float32, otype=np.complex64, batch=1, stream=0, mode=fft.FFTPlan.MODE_FFTW_ALL)
# print(img.shape)
# f = fft.FFTPlan(img_shape, np.complex64, np.complex64, 1, 0, fft.FFTPlan.MODE_FFTW_PADDING)

from pyculib.fft.binding import Plan, CUFFT_C2C
from pyculib import blas as cublas
n = (128*10)**2
data1 = np.arange(n, dtype=np.complex64).reshape(2, n//2)
data = np.arange(n, dtype=np.complex64)
orig = data.copy()
d_data = cuda.to_device(data)
#s0 = cuda.stream()
# cuda.select_device(1)
# d_data1 = cuda.to_device(data)
#s1 = cuda.stream()
# fftplan = Plan.one(CUFFT_C2C, *data.shape)
# Plan.many()
fftplan1 = Plan.many(data.shape, CUFFT_C2C, 1500)
b = cublas.Blas()
rounds = 10000
start = time.clock()


for x in range(rounds):
    # fft.fft_inplace(img)
    # cuda.select_device(0)
    # fftplan1.forward(d_data, d_data)
    # fftplan1.inverse(d_data, d_data)
    # cuda.select_device(1)
    # fftplan1.forward(d_data1, d_data1)
    #fftplan1.forward(d_data1, d_data1)
    # fftplan.inverse(d_data, d_data)
        # d_data = cuda.to_device(data)
        # cublas.dot(d_data, d_data)
    fftplan1.forward(d_data, d_data)
    fftplan1.inverse(d_data, d_data)
    # for y in range(9):
    #     # fftplan1.forward(d_data, d_data)
    #     # fftplan1.inverse(d_data, d_data)
    #     # b.dot(d_data, d_data)
    #     #b.dot(d_data1, d_data1)
    #     pass

stop = time.clock()

print(stop-start)

start = time.clock()


# for x in range(rounds):
#     # fft.fft_inplace(img)
#     # fftplan.forward(d_data, d_data)
#     # d_data = cuda.to_device(data)
#     for y in range(8):
#         fftplan.forward(d_data, d_data)
#         fftplan.inverse(d_data, d_data)
#         pass
#
# stop = time.clock()
#
# print(stop-start)

#
# start = time.clock()
#
#
# for x in range(rounds):
#     fft.fft(img, img_out)
#
# stop = time.clock()
#
# print(stop-start)
#
#
# start = time.clock()
#
# for x in range(rounds):
#     fft.ifft(img_out, img)
#
# stop = time.clock()
#
# print(stop-start)
#
# start = time.clock()
#
# for x in range(rounds):
#     fft.ifft(fft.fft(img, img_out), img)
#
# stop = time.clock()
#
# print(stop-start)