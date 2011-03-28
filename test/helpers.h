#pragma once
#define TOL_F32 1e-5

template<class T>
T RMSE(size_t n, T* a, T* b)
{ T ssq=0.0,t;
  for(size_t i=0;i<n;++i)
  { t = b[i]-a[i];
    ssq+=t*t;
  }
  return sqrt(ssq/n);
}

template<class T>
T* zeros(size_t ndim, size_t* shape)
{ size_t i,nelem;
  nelem = shape[0];
  for(i=1;i<ndim;++i)
    nelem*=shape[i];
  T* v = new T[nelem];
  memset(v,0,nelem*sizeof(T));
}

