from __future__ import print_function, absolute_import, division
import numpy as np
import unittest
from .base import CUDATestCase
from numba import cuda
import math

class TestCuFFTLib(CUDATestCase):
    def test_lib(self):
        from pyculib.fft.binding import libcufft
        cufft = libcufft()
        self.assertNotEqual(libcufft().version, 0)


class TestCuFFTPlan(CUDATestCase):
    def test_plan1d(self):
        
        from pyculib.fft.binding import Plan, CUFFT_C2C
        n = 10
        data = np.arange(n, dtype=np.complex64)
        orig = data.copy()
        fftplan = Plan.one(CUFFT_C2C, n)
        d_data = fftplan.to_device(data)
        fftplan.forward(d_data, d_data)
        fftplan.inverse(d_data, d_data)
        d_data.copy_to_host(data)
        result = data / n
        self.assertTrue(np.allclose(orig, result.real))

    def test_plan2d(self):
        
        from pyculib.fft.binding import Plan, CUFFT_C2C
        n = 2**4
        data = np.arange(n, dtype=np.complex64).reshape(2, n//2)
        orig = data.copy()
        fftplan = Plan.two(CUFFT_C2C, *data.shape)
        d_data = fftplan.to_device(data)
        fftplan.forward(d_data, d_data)
        fftplan.inverse(d_data, d_data)
        d_data.copy_to_host(data)
        result = data / n
        self.assertTrue(np.allclose(orig, result.real))

    def test_plan3d(self):
        
        from pyculib.fft.binding import Plan, CUFFT_C2C
        n = 32
        data = np.arange(n, dtype=np.complex64).reshape(2, 2, 8)

        orig = data.copy()
        fftplan = Plan.three(CUFFT_C2C, *data.shape)
        d_data = fftplan.to_device(data)
        fftplan.forward(d_data, d_data)
        fftplan.inverse(d_data, d_data)
        d_data.copy_to_host(data)
        result = data / n
        self.assertTrue(np.allclose(orig, result.real))

    def test_against_fft_1d(self):
        from pyculib.fft.binding import Plan, CUFFT_R2C     
        N = 128
        x = np.arange(N, dtype=np.float32)
          
        xf = np.fft.fft(x)
        
        halfZ = N//2 + 1
        plan = Plan.many(x.shape, CUFFT_R2C)
        d_x_gpu = plan.to_device(x)
        xf_gpu = np.zeros( halfZ, dtype=np.complex64)
        d_xf_gpu = plan.to_device(xf_gpu)
        plan.forward(d_x_gpu, d_xf_gpu)        
        d_xf_gpu.copy_to_host(xf_gpu)

        self.assertTrue( np.allclose(xf[0:halfZ], xf_gpu, atol=1e-6) )

    def test_against_fft_2d(self):
        from pyculib.fft.binding import Plan, CUFFT_R2C
        rank = 2
        rowsize = 128
        N = rank * rowsize
        x = np.arange(N, dtype=np.float32).reshape(rank, rowsize)

        xf = np.fft.fft2(x)

        halfZ = rowsize//2 + 1
        plan = Plan.many(x.shape, CUFFT_R2C)
        d_x_gpu = plan.to_device(x)
        xf_gpu = np.zeros(shape=(rank, halfZ), dtype=np.complex64)
        d_xf_gpu = plan.to_device(xf_gpu)
        plan.forward(d_x_gpu, d_xf_gpu)
        d_xf_gpu.copy_to_host(xf_gpu)
                
        self.assertTrue(np.allclose(xf[:, 0:halfZ], xf_gpu, atol=1e-6))

    def test_against_fft_3d(self):    
        from pyculib.fft.binding import Plan, CUFFT_R2C
        depth = 2
        colsize = 2
        rowsize = 64
        N = depth * colsize * rowsize
        x = np.arange(N, dtype=np.float32).reshape(depth, colsize, rowsize)
                
        xf = np.fft.fftn(x)
        
        halfZ = rowsize//2 + 1
        plan = Plan.many(x.shape, CUFFT_R2C)
        d_x_gpu = plan.to_device(x)
        xf_gpu = np.zeros(shape=(depth, colsize, halfZ), dtype=np.complex64)
        d_xf_gpu = plan.to_device(xf_gpu)
        plan.forward(d_x_gpu, d_xf_gpu)
        d_xf_gpu.copy_to_host(xf_gpu)

        self.assertTrue(np.allclose(xf[:, :, 0:halfZ], xf_gpu, atol=1e-6))

    #FFT 1D Multi-GPU Not supported via cufft Xt Specification    
    def test_against_fft_1d_mgpu(self):
        return True
        
        from pyculib.fft.binding import Plan, CUFFT_R2C, CUFFT_C2R     
        N = 32
        x = np.arange(N, dtype=np.float32)

        halfZ = N//2 + 1
        xh = np.arange(halfZ, dtype=np.complex64)
        for i in range( halfZ - 1) :
          r = x[2*i]
          if 2*i+1 < N :
              imag = x[2*i+1]
          else :
              imag = 0
          xh[i]= np.complex(r,imag)
        xh[halfZ - 1] = 0
        
        print (x)
        print (xh)
       
        xf = np.fft.fft(x)
       
        plan = Plan.many([N], CUFFT_R2C, 1, 2)
        d_x_gpu = plan.to_device(xh)
        xf_gpu = np.zeros(halfZ, dtype=np.complex64)
        #d_xf_gpu = plan.to_device(xf_gpu)
        plan.forward(d_x_gpu, d_x_gpu) 
        d_x_gpu.copy_to_host(xf_gpu)

        self.assertTrue( np.allclose(xf[0:halfZ], xf_gpu, atol=1e-6) )

    def test_against_fft_2d_mgpu(self):    
        from pyculib.fft.binding import Plan, CUFFT_R2C
        rank = 64
        rowsize = 64
        N = rank * rowsize
        x = np.arange(N, dtype=np.float32)

        halfZ = rowsize//2 + 1
        xh = np.arange(rank*halfZ, dtype=np.complex64)
        for j in range(rank) :
          for i in range( halfZ - 1) :
            ii = j*rowsize + 2*i
            r = x[ii]
            if ii+1 < N :
                imag = x[ii+1]
            else :
                imag = 0
            xh[j*halfZ + i]= np.complex(r,imag)
          xh[j*halfZ+halfZ-1] = 0
        
        x = x.reshape(rank, rowsize)
        xh = xh.reshape(rank, halfZ)

        xf = np.fft.fft2(x)

        plan = Plan.many([rank, rowsize], CUFFT_R2C, 1, 2)
        d_x_gpu = plan.to_device(xh)
        xf_gpu = np.zeros(shape=(rank, halfZ), dtype=np.complex64)
        #d_xf_gpu = plan.to_device(xf_gpu)
        plan.forward(d_x_gpu, d_x_gpu) #Inplace
        d_x_gpu.copy_to_host(xf_gpu)
                        
        self.assertTrue(np.allclose(xf[:, 0:halfZ], xf_gpu, atol=1e-6))


    def test_against_fft_3d_mgpu(self):
        from pyculib.fft.binding import Plan, CUFFT_R2C
        depth = 32
        colsize = 32
        rowsize = 32
        N = depth * colsize * rowsize
        x = np.arange(N, dtype=np.float32)
        
        halfZ = rowsize //2 + 1
        xh = np.arange(depth*colsize * halfZ, dtype=np.complex64)
        for k in range(depth) :
          for j in range(colsize):
            for i in range( halfZ-1) :
              ii = k*colsize*rowsize+j*rowsize + 2*i
              r = x[ii]
              if ii+1 < N :
                  imag = x[ii+1]
              else :
                  imag = 0
              xh[k*colsize*halfZ + j*halfZ + i]= np.complex(r,imag)
            
            xh[k*colsize*halfZ + j*halfZ + halfZ-1] = 0
          
        x = x.reshape(depth, colsize, rowsize)
        xh = xh.reshape(depth, colsize, halfZ)
                
        xf = np.fft.fftn(x)
        
        plan = Plan.many([depth, colsize, rowsize], CUFFT_R2C, 1, 2)
        d_x_gpu = plan.to_device(xh)
        xf_gpu = np.zeros(shape=(depth, colsize, halfZ), dtype=np.complex64)
        #d_xf_gpu = plan.to_device(xf_gpu)
        plan.forward(d_x_gpu, d_x_gpu)
        d_x_gpu.copy_to_host(xf_gpu)
        
        self.assertTrue(np.allclose(xf[:, :, 0:halfZ], xf_gpu, atol=1e-6))


class TestCuFFTAPI(CUDATestCase):
    def test_fft_1d_single(self):
        from pyculib.fft import fft
        N = 32
        x = np.asarray(np.arange(N), dtype=np.float32)
        xf = np.fft.fft(x)

        xf_gpu = np.empty(shape=N//2 + 1, dtype=np.complex64)
        fft(x, xf_gpu)

        self.assertTrue( np.allclose(xf[0:N//2+1], xf_gpu, atol=1e-6) )

    def test_fft_1d_double(self):
        from pyculib.fft import fft
        N = 32
        x = np.asarray(np.arange(N), dtype=np.float64)
        xf = np.fft.fft(x)

        xf_gpu = np.zeros(shape=N//2 + 1, dtype=np.complex128)
        fft(x, xf_gpu)

        self.assertTrue( np.allclose(xf[0:N//2+1], xf_gpu, atol=1e-6) )

    def test_fft_2d_single(self):
        from pyculib.fft import fft
        N2 = 2
        N1 = 32
        N = N1 * N2
        x = np.asarray(np.arange(N), dtype=np.float32).reshape(N2, N1)
        xf = np.fft.fft2(x)

        xf_gpu = np.empty(shape=(N2, N1//2 + 1), dtype=np.complex64)
        fft(x, xf_gpu)

        self.assertTrue( np.allclose(xf[:, 0:N1//2+1], xf_gpu, atol=1e-6) )

    def test_fft_2d_single_col_major(self):
        from pyculib.fft import fft
        N2 = 2
        N1 = 8
        N = N1 * N2
        x = np.asarray(np.arange(N), dtype=np.float32).reshape((N2, N1), order='F')
        xf_ref = np.fft.rfft2(x)

        xf = np.empty(shape=(N2, N1//2 + 1), dtype=np.complex64, order='F')
        fft(x, xf)
        self.assertTrue( np.allclose(xf_ref, xf, atol=1e-6) )

    def test_invfft_2d_single_col_major(self):
        from pyculib.fft import ifft
        N2 = 4
        N1 = 16
        N = (N1//2 + 1) * N2
        x = np.asarray(np.arange(N), dtype=np.complex64).reshape((N2, N1//2 + 1), order='F')
        xf_ref = np.fft.irfft2(x)
        xf = np.empty(shape=(N2, N1), dtype=np.float32, order='F')
        ifft(x, xf)
        # Note the different normalization conventions in np.fft and cuda.fft !
        xf /= N1*N2
        self.assertTrue(np.allclose(xf_ref, xf, atol=1e-6))

    def test_fft_2d_double(self):
        from pyculib.fft import fft
        N2 = 2
        N1 = 32
        N = N1 * N2
        x = np.asarray(np.arange(N), dtype=np.float64).reshape(N2, N1)
        xf = np.fft.fft2(x)

        xf_gpu = np.empty(shape=(N2, N1//2 + 1), dtype=np.complex128)
        fft(x, xf_gpu)

        self.assertTrue( np.allclose(xf[:, 0:N1//2+1], xf_gpu, atol=1e-6) )

    def test_fft_3d_single(self):
        from pyculib.fft import fft
        N3 = 2
        N2 = 2
        N1 = 32
        N = N1 * N2 * N3
        x = np.asarray(np.arange(N), dtype=np.float32).reshape(N3, N2, N1)
        xf = np.fft.fftn(x)

        xf_gpu = np.empty(shape=(N3, N2, N1//2 + 1), dtype=np.complex64)
        fft(x, xf_gpu)

        self.assertTrue( np.allclose(xf[:, :, 0:N1//2+1], xf_gpu, atol=1e-6) )

    def test_fft_3d_double(self):
        from pyculib.fft import fft
        N3 = 2
        N2 = 2
        N1 = 32
        N = N1 * N2 * N3
        x = np.asarray(np.arange(N), dtype=np.float64).reshape(N3, N2, N1)
        xf = np.fft.fftn(x)

        xf_gpu = np.empty(shape=(N3, N2, N1//2 + 1), dtype=np.complex128)
        fft(x, xf_gpu)

        self.assertTrue( np.allclose(xf[:, :, 0:N1//2+1], xf_gpu, atol=1e-6) )

    def test_fft_1d_roundtrip_single(self):
        from pyculib.fft import fft, ifft
        N = 32
        x = np.asarray(np.arange(N), dtype=np.float32)
        x0 = x.copy()
        xf_gpu = np.empty(shape=N//2 + 1, dtype=np.complex64)
        fft(x, xf_gpu)
        ifft(xf_gpu, x)
        
        #CUDA 10 uses 64bit API internally, so the percision of the numbers is higher than in the CPU version
        #Round down to match precision
        x = x.round(4)
        
        self.assertTrue( np.allclose(x / N, x0, atol=1e-6) )

    def test_fft_1d_roundtrip_double(self):
        from pyculib.fft import fft, ifft
        N = 32
        x = np.asarray(np.arange(N), dtype=np.float64)
        x0 = x.copy()
        xf_gpu = np.empty(shape=N//2 + 1, dtype=np.complex128)
        fft(x, xf_gpu)
        ifft(xf_gpu, x)
      
        self.assertTrue( np.allclose(x / N, x0, atol=1e-6) )

    def test_fft_2d_roundtrip_single(self):
        from pyculib.fft import fft, ifft
        N2 = 2
        N1 = 32
        N = N2 * N1
        x = np.asarray(np.arange(N), dtype=np.float32).reshape(N2, N1)
        x0 = x.copy()
        xf_gpu = np.empty(shape=(N2, N1//2 + 1), dtype=np.complex64)
        fft(x, xf_gpu)
        ifft(xf_gpu, x)
                
        self.assertTrue( np.allclose(x / N, x0, atol=1e-6) )

    def test_fft_3d_roundtrip_single(self):
        from pyculib.fft import fft, ifft
        N3 = 2
        N2 = 2
        N1 = 32
        N = N3 * N2 * N1
        x = np.asarray(np.arange(N), dtype=np.float32).reshape(N3, N2, N1)
        x0 = x.copy()
        xf_gpu = np.empty(shape=(N3, N2, N1//2 + 1), dtype=np.complex64)
        fft(x, xf_gpu)
        ifft(xf_gpu, x)
        self.assertTrue( np.allclose(x / N, x0, atol=1e-6) )

    def test_fft_inplace_1d_single(self):
        from pyculib.fft import fft_inplace
        N = 32
        x = np.asarray(np.arange(N), dtype=np.complex64)
        xf = np.fft.fft(x)

        fft_inplace(x)

        self.assertTrue( np.allclose(xf, x, atol=1e-6) )

    def test_fft_inplace_1d_double(self):
        from pyculib.fft import fft_inplace
        N = 32
        x = np.asarray(np.arange(N), dtype=np.complex128)
        xf = np.fft.fft(x)

        fft_inplace(x)

        self.assertTrue( np.allclose(xf, x, atol=1e-6) )

    def test_fft_inplace_2d_single(self):
        from pyculib.fft import fft_inplace
        N1 = 32
        N2 = 2
        N = N1 * N2
        x = np.asarray(np.arange(N), dtype=np.complex64).reshape(N2, N1)
        xf = np.fft.fft2(x)

        fft_inplace(x)

        self.assertTrue( np.allclose(xf, x, atol=1e-6) )

    def test_fft_inplace_2d_double(self):
        from pyculib.fft import fft_inplace
        N1 = 32
        N2 = 2
        N = N1 * N2
        x = np.asarray(np.arange(N), dtype=np.complex128).reshape(N2, N1)
        xf = np.fft.fft2(x)

        fft_inplace(x)

        self.assertTrue( np.allclose(xf, x, atol=1e-6) )

    def test_fft_1d_roundtrip_single_2(self):
        from pyculib.fft import fft_inplace, ifft_inplace
        N = 32
        x = np.asarray(np.arange(N), dtype=np.complex64)
        x0 = x.copy()

        fft_inplace(x)
        ifft_inplace(x)

        self.assertTrue( np.allclose(x / N, x0, atol=1e-6) )

    def test_fft_1d_roundtrip_double_2(self):
        from pyculib.fft import fft_inplace, ifft_inplace
        N = 32
        x = np.asarray(np.arange(N), dtype=np.complex128)
        x0 = x.copy()

        fft_inplace(x)
        ifft_inplace(x)

        self.assertTrue( np.allclose(x / N, x0, atol=1e-6) )


    def test_fft_2d_roundtrip_single_2(self):
        from pyculib.fft import fft_inplace, ifft_inplace
        N2 = 2
        N1 = 32
        N = N1 * N2
        x = np.asarray(np.arange(N), dtype=np.complex64).reshape(N2, N1)
        x0 = x.copy()

        fft_inplace(x)
        ifft_inplace(x)

        self.assertTrue( np.allclose(x / N, x0, atol=1e-6) )

    def test_fft_3d_roundtrip_double(self):
        from pyculib.fft import fft_inplace, ifft_inplace
        N3 = 2
        N2 = 2
        N1 = 8
        N = N3 * N2 * N1
        x = np.asarray(np.arange(N), dtype=np.complex128).reshape(N3, N2, N1)
        x0 = x.copy()

        fft_inplace(x)
        ifft_inplace(x)

        self.assertTrue( np.allclose(x / N, x0, atol=1e-6) )


def test():
    import sys
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    mod = sys.modules[__name__]
    for name in dir(mod):
        if name.startswith('Test'):
            test_class = getattr(mod, name)
            tests = loader.loadTestsFromTestCase(test_class)
            suite.addTests(tests)
    runner = unittest.runner.TextTestRunner()
    return runner.run(suite)


test_cases = (TestCuFFTLib, TestCuFFTPlan, TestCuFFTAPI)

if __name__ == '__main__':
    unittest.main()

