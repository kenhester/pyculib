from __future__ import absolute_import
import numpy as np
from ctypes import c_void_p, c_int, c_int64, c_size_t, c_longlong, POINTER, byref
from numba import cuda

from numba.cuda.cudadrv.drvapi import cu_stream
from numba.cuda.cudadrv.driver import device_pointer
from pyculib.utils import Lib, ctype_function, finalizer

STATUS = {
    0x0: 'CUFFT_SUCCESS',
    0x1: 'CUFFT_INVALID_PLAN',
    0x2: 'CUFFT_ALLOC_FAILED',
    0x3: 'CUFFT_INVALID_TYPE',
    0x4: 'CUFFT_INVALID_VALUE',
    0x5: 'CUFFT_INTERNAL_ERROR',
    0x6: 'CUFFT_EXEC_FAILED',
    0x7: 'CUFFT_SETUP_FAILED',
    0x8: 'CUFFT_INVALID_SIZE',
    0x9: 'CUFFT_UNALIGNED_DATA',
}

cufftResult = c_int

# Trandorm Direction
CUFFT_FORWARD = -1
CUFFT_INVERSE = 1

# cufftType
CUFFT_R2C = 0x2a  # Real to Complex (interleaved)
CUFFT_C2R = 0x2c  # Complex (interleaved) to Real
CUFFT_C2C = 0x29  # Complex to Complex, interleaved
CUFFT_D2Z = 0x6a  # Double to Double-Complex
CUFFT_Z2D = 0x6c  # Double-Complex to Double
CUFFT_Z2Z = 0x69  # Double-Complex to Double-Complex

cufftType = c_int

CUFFT_COMPATIBILITY_NATIVE = 0x00
CUFFT_COMPATIBILITY_FFTW_PADDING = 0x01  # The default value
CUFFT_COMPATIBILITY_FFTW_ASYMMETRIC = 0x02
CUFFT_COMPATIBILITY_FFTW_ALL = 0x03

CUFFT_COMPATIBILITY_DEFAULT = CUFFT_COMPATIBILITY_FFTW_PADDING

cufftCompatibility = c_int

cufftHandle = c_int

#multi-gpu
# cufftXtSubFormat identifies the data layout of
# a memory descriptor owned by cufft.
# note that multi GPU cufft does not yet support out-of-place transforms
CUFFT_XT_FORMAT_INPUT = 0x00              #by default input is in linear order across GPUs
CUFFT_XT_FORMAT_OUTPUT = 0x01             #by default output is in scrambled order depending on transform
CUFFT_XT_FORMAT_INPLACE = 0x02            #by default inplace is input order, which is linear across GPUs
CUFFT_XT_FORMAT_INPLACE_SHUFFLED = 0x03   #shuffled output order after execution of the transform
CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED = 0x04  #shuffled input order prior to execution of 1D transforms
CUFFT_FORMAT_UNDEFINED = 0x05

cufftXtSubFormat = c_int
  
# cufftXtWorkAreaPolicy specifies policy for cufftXtSetWorkAreaPolicy
CUFFT_WORKAREA_MINIMAL = 0 # maximum reduction
CUFFT_WORKAREA_USER = 1 # use workSize parameter as limit
CUFFT_WORKAREA_PERFORMANCE = 2 # default - 1x overhead or more, maximum performance

#cufftXtWorkAreaPolicy = c_int

# cufftXtCopyType specifies the type of copy for cufftXtMemcpy
CUFFT_COPY_HOST_TO_DEVICE = 0x00
CUFFT_COPY_DEVICE_TO_HOST = 0x01
CUFFT_COPY_DEVICE_TO_DEVICE: 0x02
CUFFT_COPY_UNDEFINED: 0x03
        
cufftXtCopyType = c_int

cudaDataType = c_int

class CuFFTError(Exception):
    def __init__(self, code):
        super(CuFFTError, self).__init__(STATUS[code])


class libcufft(Lib):
    lib = 'cufft'
    ErrorType = CuFFTError

    @property
    def version(self):
        ver = c_int(0)
        self.cufftGetVersion(byref(ver))
        return ver.value

    cufftGetVersion = ctype_function(cufftResult, POINTER(c_int))

    cufftPlan1d = ctype_function(cufftResult,
                                 POINTER(cufftHandle),  # plan
                                 c_int,  # nx
                                 cufftType,  # type
                                 c_int,
                                 # batch - deprecated - use cufftPlanMany
    )

    cufftPlan2d = ctype_function(cufftResult,
                                 POINTER(cufftHandle),  # plan
                                 c_int,  # nx
                                 c_int,  # ny
                                 cufftType  # type
    )

    cufftPlan3d = ctype_function(cufftResult,
                                 POINTER(cufftHandle),  # plan
                                 c_int,  # nx
                                 c_int,  # ny
                                 c_int,  # nz
                                 cufftType  # type
    )

    
    #cufftResult CUFFTAPI cufftCreate(cufftHandle * handle);
    cufftCreate = ctype_function(cufftResult,
                                  POINTER(cufftHandle),  # plan
    )
   
    #cufftResult CUFFTAPI cufftMakePlanMany(cufftHandle plan,
    #                                       int rank,
    #                                       int *n,
    #                                       int *inembed, int istride, int idist,
    #                                       int *onembed, int ostride, int odist,
    #                                       cufftType type,
    #                                       int batch,
    #                                       size_t *workSize);
    cufftMakePlanMany = ctype_function(cufftResult,
                                 cufftHandle,  # plan
                                 c_int,  # rank
                                 POINTER(c_int),  # n
                                 POINTER(c_int),  # inembed
                                 c_int,  # inembed
                                 c_int,  # idist
                                 POINTER(c_int),  # onembed
                                 c_int,  # ostride
                                 c_int,  # odist
                                 cufftType,  # type
                                 c_int, # batch
                                 POINTER(c_size_t), # workSize
    )
   
    #cufftResult CUFFTAPI cufftMakePlanMany64(cufftHandle plan,
    #                                         int rank,
    #                                         long long int *n,
    #                                         long long int *inembed,
    #                                         long long int istride,
    #                                         long long int idist,
    #                                         long long int *onembed,
    #                                         long long int ostride, long long int odist,
    #                                         cufftType type,
    #                                         long long int batch,
    #                                         size_t * workSize);
    cufftMakePlanMany64 = ctype_function(cufftResult,
                                 cufftHandle,  # plan
                                 c_int,  # rank
                                 POINTER(c_longlong),  # n
                                 POINTER(c_longlong),  # inembed
                                 c_longlong,  # inembed
                                 c_longlong,  # idist
                                 POINTER(c_longlong),  # onembed
                                 c_longlong,  # ostride
                                 c_longlong,  # odist
                                 cufftType,  # type
                                 c_longlong, # batch
                                 POINTER(c_size_t), # orkSize
    )

    cufftPlanMany = ctype_function(cufftResult,
                                   POINTER(cufftHandle),  # plan
                                   c_int,  # rank
                                   c_void_p,  # POINTER(c_int) n
                                   c_void_p,  # POINTER(c_int) inembed
                                   c_int,  # istride
                                   c_int,  # idist
                                   c_void_p,  # POINTER(c_int) onembed
                                   c_int,  # ostride
                                   c_int,  # odist
                                   cufftType,  # type
                                   c_int,  # batch
    )

    cufftDestroy = ctype_function(cufftResult,
                                  cufftHandle,  # plan
    )

    cufftExecC2C = ctype_function(cufftResult,
                                  cufftHandle,  # plan
                                  c_void_p,  # POINTER(cufftComplex) idata
                                  c_void_p,  # POINTER(cufftComplex) odata
                                  c_int,  # direction
    )

    cufftExecR2C = ctype_function(cufftResult,
                                  cufftHandle,  # plan
                                  c_void_p,  # POINTER(cufftReal) idata
                                  c_void_p,  # POINTER(cufftComplex) odata
                                  c_int,
    )

    cufftExecC2R = ctype_function(cufftResult,
                                  cufftHandle,  # plan
                                  c_void_p,  # POINTER(cufftComplex) idata
                                  c_void_p,  # POINTER(cufftReal) odata
                                  c_int,
    )

    cufftExecZ2Z = ctype_function(cufftResult,
                                  cufftHandle,  # plan
                                  c_void_p,  # POINTER(cufftDoubleComplex) idata
                                  c_void_p,  # POINTER(cufftDoubleComplex) odata
                                  c_int,  # direction
    )

    cufftExecD2Z = ctype_function(cufftResult,
                                  cufftHandle,  # plan
                                  c_void_p,  # POINTER(cufftDoubleReal) idata
                                  c_void_p,  # POINTER(cufftDoubleComplex) odata
                                  c_int,
    )

    cufftExecZ2D = ctype_function(cufftResult,
                                  cufftHandle,  # plan
                                  c_void_p,  # POINTER(cufftDoubleComplex) idata
                                  c_void_p,  # POINTER(cufftDoubleReal) odata
                                  c_int,
    )

    cufftSetStream = ctype_function(cufftResult,
                                    cufftHandle,  # plan,
                                    cu_stream,  # stream
    )

    cufftSetCompatibilityMode = ctype_function(cufftResult,
                                               cufftHandle,  # plan,
                                               cufftCompatibility  # mode
    )

    # multi-GPU routines
    
    #cufftResult CUFFTAPI cufftXtSetGPUs(cufftHandle handle, int nGPUs, int *whichGPUs);
    cufftXtSetGPUs = ctype_function(cufftResult,
                                               cufftHandle,  # plan,
                                               c_int,  # nGPUs
                                               POINTER(c_int) #whichGPUs)
    )
    
    #cufftResult CUFFTAPI cufftXtMalloc(cufftHandle plan,
    #                                   cudaLibXtDesc ** descriptor,
    #                                   cufftXtSubFormat format);
    cufftXtMalloc = ctype_function(cufftResult,
                                               cufftHandle,  # plan,
                                               POINTER(c_void_p),  # descriptor
                                               cufftXtSubFormat #format
    )
    
    #cufftResult CUFFTAPI cufftXtMemcpy(cufftHandle plan,
    #                                   void *dstPointer,
    #                                   void *srcPointer,
    #                                   cufftXtCopyType type);
    cufftXtMemcpy = ctype_function(cufftResult,
                                               cufftHandle,  # plan,
                                               c_void_p,  # dstPointer
                                               c_void_p, #srcPointer
                                               cufftXtCopyType #type
    )                                           
   
    #cufftResult CUFFTAPI cufftXtFree(cudaLibXtDesc *descriptor);
    cufftXtFree = ctype_function(cufftResult,
                                               c_void_p  # POINTER(descriptor)
    )                                           
    
    #cufftResult CUFFTAPI cufftXtSetWorkArea(cufftHandle plan, void **workArea);
    cufftXtSetWorkArea = ctype_function(cufftResult,
                                               cufftHandle,  # plan,
                                               POINTER(c_void_p)  # POINTER(workArea)
    )                                           
  
    #cufftResult CUFFTAPI cufftXtExecDescriptorC2C(cufftHandle plan,
    #                                              cudaLibXtDesc *input,
    #                                              cudaLibXtDesc *output,
    #                                              int direction);
    cufftXtExecDescriptorC2C = ctype_function(cufftResult,
                                               cufftHandle,  # plan,
                                               c_void_p,  # POINTER(input)
                                               c_void_p,  # POINTER(output)
                                               c_int #direction
    )                                           
   
    #cufftResult CUFFTAPI cufftXtExecDescriptorR2C(cufftHandle plan,
    #                                              cudaLibXtDesc *input,
    #                                              cudaLibXtDesc *output);
    cufftXtExecDescriptorR2C = ctype_function(cufftResult,
                                               cufftHandle,  # plan,
                                               c_void_p,  # POINTER(input)
                                               c_void_p   # POINTER(output)
    )  
    #cufftResult CUFFTAPI cufftXtExecDescriptorC2R(cufftHandle plan,
    #                                              cudaLibXtDesc *input,
    #                                              cudaLibXtDesc *output);
    cufftXtExecDescriptorC2R = ctype_function(cufftResult,
                                               cufftHandle,  # plan,
                                               c_void_p,  # POINTER(input)
                                               c_void_p  # POINTER(output)
    )
    
    #cufftResult CUFFTAPI cufftXtExecDescriptorZ2Z(cufftHandle plan,
    #                                              cudaLibXtDesc *input,
    #                                              cudaLibXtDesc *output,
    #                                              int direction);
    cufftXtExecDescriptorZ2Z = ctype_function(cufftResult,
                                               cufftHandle,  # plan,
                                               c_void_p,  # POINTER(input)
                                               c_void_p,  # POINTER(output)
                                               c_int #direction
    )                                           
  
    #cufftResult CUFFTAPI cufftXtExecDescriptorD2Z(cufftHandle plan,
    #                                              cudaLibXtDesc *input,
    #                                              cudaLibXtDesc *output);
    cufftXtExecDescriptorD2Z = ctype_function(cufftResult,
                                               cufftHandle,  # plan,
                                               c_void_p,  # POINTER(input)
                                               c_void_p   # POINTER(output)
    )  
   
    #cufftResult CUFFTAPI cufftXtExecDescriptorZ2D(cufftHandle plan,
    #                                              cudaLibXtDesc *input,
    #                                              cudaLibXtDesc *output);
    cufftXtExecDescriptorZ2D = ctype_function(cufftResult,
                                               cufftHandle,  # plan,
                                               c_void_p,  # POINTER(input)
                                               c_void_p   # POINTER(output)
    )  

    #cufftResult CUFFTAPI cufftXtMakePlanMany(cufftHandle plan,
    #                                     int rank,
    #                                     long long int *n,
    #                                     long long int *inembed,
    #                                     long long int istride,
    #                                     long long int idist,
    #                                     cudaDataType inputtype,
    #                                     long long int *onembed,
    #                                     long long int ostride,
    #                                     long long int odist,
    #                                     cudaDataType outputtype,
    #                                     long long int batch,
    #                                     size_t *workSize,
    #                                     cudaDataType executiontype);
    cufftXtMakePlanMany = ctype_function(cufftResult,
                                 cufftHandle,  # plan
                                 c_int,  # rank
                                 POINTER(c_longlong),  # n
                                 POINTER(c_longlong),  # inembed
                                 c_longlong,  # inembed
                                 c_longlong,  # idist
                                 cudaDataType, # inputtype
                                 POINTER(c_longlong),  # onembed
                                 c_longlong,  # ostride
                                 c_longlong,  # odist
                                 cudaDataType, # outputtype
                                 c_longlong, # batch
                                 POINTER(c_size_t), # workSize
                                 cudaDataType, # executiontype
    )


    #cufftResult CUFFTAPI cufftXtGetSizeMany(cufftHandle plan,
    #                                    int rank,
    #                                    long long int *n,
    #                                    long long int *inembed,
    #                                    long long int istride,
    #                                    long long int idist,
    #                                    cudaDataType inputtype,
    #                                    long long int *onembed,
    #                                    long long int ostride,
    #                                    long long int odist,
    #                                    cudaDataType outputtype,
    #                                    long long int batch,
    #                                    size_t *workSize,
    #                                    cudaDataType executiontype);
    cufftXtGetSizeMany = ctype_function(cufftResult,
                                 cufftHandle,  # plan
                                 c_int,  # rank
                                 POINTER(c_longlong),  # n
                                 POINTER(c_longlong),  # inembed
                                 c_longlong,  # inembed
                                 c_longlong,  # idist
                                 cudaDataType, # inputtype
                                 POINTER(c_longlong),  # onembed
                                 c_longlong,  # ostride
                                 c_longlong,  # odist
                                 cudaDataType, # outputtype
                                 c_longlong, # batch
                                 POINTER(c_size_t), # workSize
                                 cudaDataType, # executiontype
    )

    #cufftResult CUFFTAPI cufftXtExec(cufftHandle plan,
    #                             void *input,
    #                             void *output,
    #                             int direction);
    cufftXtExec = ctype_function(cufftResult,
                                 cufftHandle,  # plan
                                 c_void_p,  # input
                                 c_void_p, # output
                                 c_int, #direction
    )

    #cufftResult CUFFTAPI cufftXtExecDescriptor(cufftHandle plan,
    #                                       cudaLibXtDesc *input,
    #                                       cudaLibXtDesc *output,
    #                                       int direction);
    cufftXtExecDescriptor = ctype_function(cufftResult,
                                 cufftHandle,  # plan
                                 c_void_p,  # input
                                 c_void_p, # output
                                 c_int, #direction
    )

    #cufftResult CUFFTAPI cufftXtSetWorkAreaPolicy(cufftHandle plan, cufftXtWorkAreaPolicy policy, size_t *workSize);
    #cufftXtSetWorkAreaPolicy = ctype_function(cufftResult,
    #                             cufftHandle,  # plan
    #                             cufftXtWorkAreaPolicy,  # policy
    #                             POINTER(c_size_t), # workSize
    #)


cufft_dtype_to_name = {
    CUFFT_R2C: 'R2C',
    CUFFT_C2R: 'C2R',
    CUFFT_C2C: 'C2C',
    CUFFT_D2Z: 'D2Z',
    CUFFT_Z2D: 'Z2D',
    CUFFT_Z2Z: 'Z2Z',
}

class PlanDataHelper(finalizer.OwnerMixin) :
    def __init__(self):
       self._d_data = c_int(0)
       self._ngpu = 1
    
    def __del__(self):
      if self._ngpu > 1 :
        if self._d_data != 0 :
           self._api.cufftXtFree(self._d_data)
           self._d_data = 0

          
    @classmethod
    def to_device(cls, plan, data, stream=None):
        inst = object.__new__(cls)
        inst._ngpu = plan._ngpu
        inst._api = libcufft()
        "copy host memory to device"
        if plan._ngpu <= 1:
          inst._d_data = cuda.to_device(data, stream)
        else:
          d_data = c_int()
          if stream != None : 
              inst._api.cufftSetStream(plan._handle, stream.handle)
          inst._api.cufftXtMalloc(plan._handle, byref(d_data), CUFFT_XT_FORMAT_INPLACE)
          inst._api.cufftXtMemcpy(plan._handle, d_data, data, CUFFT_COPY_HOST_TO_DEVICE)
          inst._d_data = d_data

        return inst; 

    def copy_to_host(self, data):
        "copy device memory to host"
        if self._ngpu <= 1 :
          self._d_data.copy_to_host(data)
        else :
          self._api.cufftXtMemcpy(plan, data, self._d_data, CUFFT_COPY_DEVICE_TO_HOST) 

class Plan(finalizer.OwnerMixin):
    @classmethod
    def one(cls, dtype, nx, ngpu=1):
        "cufftPlan1d"
        inst = object.__new__(cls)
        inst._api = libcufft()
        inst._handle = cufftHandle()
        inst._ngpu = ngpu
        
        BATCH = 1  # deprecated args to cufftPlan1d
        inst._api.cufftPlan1d(byref(inst._handle), int(nx), int(dtype),
                              BATCH)
        inst.dtype = dtype
        inst._finalizer_track((inst._handle, inst._api))
        return inst

    @classmethod
    def two(cls, dtype, nx, ny, ngpu=1):
        "cufftPlan2d"
        inst = object.__new__(cls)
        inst._api = libcufft()
        inst._handle = cufftHandle()
        inst._ngpu = ngpu

        inst._api.cufftPlan2d(byref(inst._handle), int(nx), int(ny),
                              int(dtype))
        inst.dtype = dtype
        inst._finalizer_track((inst._handle, inst._api))
        return inst

    @classmethod
    def three(cls, dtype, nx, ny, nz, ngpu=1):
        "cufftPlan3d"
        inst = object.__new__(cls)
        inst._api = libcufft()
        inst._handle = cufftHandle()
        inst._ngpu = ngpu

        inst._api.cufftPlan3d(byref(inst._handle), int(nx), int(ny),
                              int(nz), int(dtype))
        inst.dtype = dtype
        inst._finalizer_track((inst._handle, inst._api))
        return inst

    @classmethod
    def many(cls, shape, dtype, batch=1, ngpu=1):
        "cufftPlanMany"
        inst = object.__new__(cls)
        inst._api = libcufft()
        inst._handle = cufftHandle()
        inst._ngpu = ngpu

        c_shape = np.asarray(shape, dtype=np.int32)
        if ngpu <= 1 :
            inst._api.cufftPlanMany(byref(inst._handle),
                                len(shape),
                                c_shape.ctypes.data,
                                None, 1, 0,
                                None, 1, 0,
                                int(dtype), int(batch))
        else :
            workSize = c_size_t()
            size=c_longlong(0)
            for element in c_shape:
                size*=element
                
            inst._api.cufftCreate(byref(inst._handle) )
            inst._api.cufftXtSetGPUs(inst._handle, ngpu, gpuid) 
            inst._api.cufftMakePlanMany64(inst._handle,
                                len(shape),
                                c_shape.ctypes.data,
                                None, 1, size,
                                None, 1, size,
                                int(dtype), int(batch), byref(workSize))

        inst.shape = shape
        inst.dtype = dtype
        inst.batch = batch
        inst._finalizer_track((inst._handle, inst._api))
        return inst

    @classmethod
    def _finalize(cls, res):
        handle, api = res
        api.cufftDestroy(handle)

    def set_stream(self, stream):
        "Associate a CUDA stream to this plan object"
        return self._api.cufftSetStream(self._handle, stream.handle)

    def to_device(self, data, stream=None):
        "copy host memory to device"
        return PlanDataHelper.to_device(self, data, stream)

        
    def set_compatibility_mode(self, mode):
        return self._api.cufftSetCompatibilityMode(self._handle, mode)

    def set_native_mode(self):
        return self.set_compatibility_mode(CUFFT_COMPATIBILITY_NATIVE)

    def set_fftw_padding_mode(self):
        return self.set_compatibility_mode(CUFFT_COMPATIBILITY_FFTW_PADDING)

    def set_fftw_asymmetric_mode(self):
        return self.set_compatibility_mode(CUFFT_COMPATIBILITY_FFTW_ASYMMETRIC)

    def set_fftw_all_mode(self):
        return self.set_compatibility_mode(CUFFT_COMPATIBILITY_FFTW_ALL)

    def exe(self, idata, odata, dir):
        postfix = cufft_dtype_to_name[self.dtype]
        if self._ngpu <= 1 :
          meth = getattr(self._api, 'cufftExec' + postfix)
        else :
          meth = getattr(self._api, 'cufftXtExecDescriptor' + postfix)
        
        if hasattr(idata,"_d_data") :  
          return meth(self._handle, device_pointer(idata._d_data),
                    device_pointer(odata._d_data), int(dir))

        return meth(self._handle, device_pointer(idata),
                    device_pointer(odata), int(dir))

    def forward(self, idata, odata):
        return self.exe(idata, odata, dir=CUFFT_FORWARD)

    def inverse(self, idata, odata):
        return self.exe(idata, odata, dir=CUFFT_INVERSE)
