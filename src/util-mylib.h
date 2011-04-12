#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <array.h>

void Haar_Transform(Array *a);
void Haar_Inverse_Transform(Array *a);

Array* ZOrder_Transform(Array *in);
Array* ZOrder_Inverse_Transform(Array *in);

#ifdef __cplusplus
} //extern "C"
#endif

