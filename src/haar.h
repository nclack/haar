/** \file 
 *  Haar transform
 *  \author Nathan Clack <clackn@janelia.hhmi.org>
 *  \date   2011
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include<stdint.h>

typedef int64_t stride_t;

typedef struct tagDomainList
{ stride_t *shapes;
  stride_t sz[2];
  stride_t cursor;
} DomainList;

typedef struct tagHaarWorkspace
{
  DomainList domains;
} HaarWorkspace;

#define HAAR_WORKSPACE_INIT {{0}}  ///< static initializer for HaarWorkspace

void HaarWorkspaceInit (HaarWorkspace* ws);
void HaarWorkspaceClean(HaarWorkspace* ws);

void haar(HaarWorkspace* ws,
          stride_t ndim, stride_t* shape,
          float* out,  stride_t* ostrides,
          float* in,   stride_t* istrides);

void ihaar(HaarWorkspace* ws,
           stride_t ndim, stride_t* shape,
           float* out,  stride_t* ostrides,
           float* in,   stride_t* istrides);

#ifdef __cplusplus
} //extern "C"
#endif
