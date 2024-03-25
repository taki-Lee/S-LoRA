import slora
import triton
import triton.language as tl
# from slora._kernels import dispatch_bgmv
import torch
import nvtx
import rpyc
import asyncio
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def async_wrap(f):
    f = rpyc.async_(f)
    async def _func(*args, **kwargs):
        ans = f(*args, **kwargs)
        await asyncio.to_thread(ans.wait)
        # raise if exception
        return ans.value
    return _func

def _matmul(A, B, stream=None):
    if stream==None:
        ans = torch.mm(A,B)
    else:
        with torch.cuda.stream(stream):
            # with nvtx.annotate("torch.matmul"):
                # B = B.to("cpu")
                # B = B.to("cuda")
            ans = torch.mm(A, B)
    return ans

async def main():
    global start
    start = time.time()
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    s3 = torch.cuda.Stream()
    # Initialise cuda tensors here. E.g.:
    print(torch.cuda.is_available())
    # exit()
    A = torch.rand(1000, 1000, device = 'cuda:0')
    B = torch.rand(1000, 1000, device = 'cuda:0')
    # Wait for the above tensors to initialise.
    torch.cuda.synchronize()

    _matmul(A,B, s1)
    _matmul(A,B, s2)
    _matmul(A,B, s1)
    _matmul(A,B, s2)
    _matmul(A,B, s1)
    _matmul(A,B, s2)
    _matmul(A,B, s1)
    _matmul(A,B, s2)
    _matmul(A,B, s1)
    _matmul(A,B, s2)

    time.sleep(0.5)
    
    a = await asyncio.gather(
        asyncio.to_thread(_matmul, A, B, s1),
        asyncio.to_thread(_matmul, A, B, s2),
        asyncio.to_thread(_matmul, A, B, s3),
        asyncio.to_thread(_matmul, A, B, s1),
        asyncio.to_thread(_matmul, A, B, s2),
        asyncio.to_thread(_matmul, A, B, s3),
        asyncio.to_thread(_matmul, A, B, s1),
        asyncio.to_thread(_matmul, A, B, s2),
        asyncio.to_thread(_matmul, A, B, s3),
        asyncio.to_thread(_matmul, A, B, s1),
    )
    print(a[0])
    # Wait for C and D to be computed.
    # torch.cuda.synchronize()
    # Do stuff with C and D.
    return a

    

if __name__ == "__main__":
    a = asyncio.run(main())
    print("main ",a)
    # main()