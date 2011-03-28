#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include<stdlib.h>

typedef struct tagDomainList
{ size_t *shapes;
  size_t sz[2];
  size_t cursor;
} DomainList;
typedef struct tagHaarWorkspace
{
  DomainList domains;
} HaarWorkspace;

#define HAAR_WORKSPACE_INIT {{0}} 

void HaarWorkspaceInit (HaarWorkspace* ws);
void HaarWorkspaceClean(HaarWorkspace* ws);

/* shape   - The size of the volume to transform
 * strides - ndim+1 element arrays
 *           describing a linear layout for the scalar field
 *           The stride between individual scalars should be stride[0].
 */

void haar(HaarWorkspace* ws,
          size_t ndim, size_t* shape,
          float* out,  size_t* ostrides,
          float* in,   size_t* istrides);

void ihaar(HaarWorkspace* ws,
           size_t ndim, size_t* shape,
           float* out,  size_t* ostrides,
           float* in,   size_t* istrides);

#ifdef __cplusplus
} //extern "C"
#endif
