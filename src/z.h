#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>

void zorder(size_t ndim, size_t *shape,
            float *out, size_t *ostrides,
            float *in , size_t *istrides);

#ifdef __cplusplus
} //extern "C" {
#endif
