#ifndef ROD_CUDA_MUTEX_H
#define ROD_CUDA_MUTEX_H

namespace rod
{

class Mutex
{
public:
    __device__ Mutex() : m_mtx(0) {}

    __device__ void lock()
    {
        while(atomicCAS(&m_mtx, 0, 1) != 0);
    }
    __device__void unlock()
    {
        atomicExch(&m_mtx, 0);
    }

    __device__ bool locked() const { return m_mtx!=0; }

private:
    unsigned int m_mtx;
};

class Lock
{
public:
    __device__ Lock(Mutex &mtx) 
        : m_mtx(mtx)
    {
        m_mtx.lock();
        m_lock_held = true;
    }

    __device__ ~Lock()
    {
        if(m_lock_held)
            m_mtx.unlock();
    }

    void unlock()
    {
        if(m_lock_held)
        {
            m_mtx.unlock();
            m_lock_held = false;
        }
    }

private:
    bool m_lock_held;
};

}

#endif
