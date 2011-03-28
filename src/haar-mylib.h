#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <array.h>

void Haar_Transform(Array *a);
void Haar_Inverse_Transform(Array *a);

#ifdef __cplusplus
} //extern "C"
#endif

