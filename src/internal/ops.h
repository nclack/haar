#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include<stdlib.h>
#include<stdint.h>

unsigned u32log2(uint32_t v);
unsigned u64log2(uint64_t v);

void copy  (size_t ndim , size_t *shape , float *x , size_t *sx , float *y , size_t *sy);
void swap  (size_t ndim , size_t *shape , float *x , size_t *sx , float *y , size_t *sy);
void add   (size_t ndim , size_t *shape , float *z , size_t *sz , float *x , size_t *sx , float *y , size_t *sy);
void sub   (size_t ndim , size_t *shape , float *z , size_t *sz , float *x , size_t *sx , float *y , size_t *sy);
void sub2  (size_t ndim , size_t *shape , float *z , size_t *sz , float *x , size_t *sx , float *y , size_t *sy);
void mul_ip(size_t ndim , size_t *shape , float *z , size_t *sz , float a);
void gather (size_t idim , size_t ndim , size_t *shape , float* z , size_t *s);
void scatter(size_t idim , size_t ndim , size_t *shape , float* z , size_t *s);
#ifdef __cplusplus
} //extern "C"
#endif
