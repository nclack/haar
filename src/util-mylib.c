#include <stdlib.h>
#include <stdio.h>

#include <array.h>

#include <haar.h>
#include <z.h>

/*
 * Helpers
 */

static inline
void compute_strides_i64(Array *a, stride_t *out)
{ size_t i;
  out[0] = 1;
  for(i=0;i<a->ndims;++i)
    out[i+1] = out[i]*(a->dims[i]);
}

static inline
void copy_dims_i64(size_t ndim, Dimn_Type *dims, stride_t *shape)
{
  size_t i;
  for(i=0;i<ndim;++i)
    shape[i] = dims[i];
}

static inline
void compute_strides_size_t(Array *a, size_t *out)
{ size_t i;
  out[0] = 1;
  for(i=0;i<a->ndims;++i)
    out[i+1] = out[i]*(a->dims[i]);
}

static inline
void copy_dims_size_t(size_t ndim, Dimn_Type *dims, size_t *shape)
{
  size_t i;
  for(i=0;i<ndim;++i)
    shape[i] = dims[i];
}

/*
 * Haar
 */

#define sizeof_strides_bytes(array) (sizeof(stride_t)*((array)->ndims+1))
#define sizeof_shapes_bytes(array)  (sizeof(stride_t)*((array)->ndims))

void Haar_Transform(Array *a)
{ stride_t *strides, *shape;
  HaarWorkspace ws = HAAR_WORKSPACE_INIT;
  if(!( strides = alloca(sizeof_strides_bytes(a)) )) goto MemoryError1;
  if(!( shape   = alloca(sizeof_shapes_bytes(a) ) )) goto MemoryError2;
  compute_strides_i64(a,strides);
  copy_dims_i64(a->ndims, a->dims, shape);
  haar(&ws,a->ndims,shape,a->data,strides,a->data,strides); //in-place
  HaarWorkspaceClean(&ws);
  return;
MemoryError1:
  fprintf(stderr,"Memory Error: %s(%d) - Failed to allocate %lu bytes on stack.\n",
      __FILE__,__LINE__,sizeof_strides_bytes(a));
  exit(1);
  
MemoryError2:
  fprintf(stderr,"Memory Error: %s(%d) - Failed to allocate %lu bytes on stack.\n",
          __FILE__,__LINE__,sizeof_shapes_bytes(a));
  exit(1);  
}

void Haar_Inverse_Transform(Array *a)
{ stride_t *strides,*shape;
  HaarWorkspace ws = {{0}};
  HaarWorkspaceInit(&ws);
  if(!( strides = alloca(sizeof_strides_bytes(a)) )) goto MemoryError1;
  if(!( shape   = alloca(sizeof_shapes_bytes(a) ) )) goto MemoryError2;
  compute_strides_i64(a,strides);
  copy_dims_i64(a->ndims, a->dims, shape);
  ihaar(&ws,a->ndims,shape,a->data,strides,a->data,strides); //in-place
  HaarWorkspaceClean(&ws);
  return;
MemoryError1:
  fprintf(stderr,"Memory Error: %s(%d) - Failed to allocate %lu bytes on stack.\n",
          __FILE__,__LINE__,sizeof_strides_bytes(a));
  exit(1);
  
MemoryError2:
  fprintf(stderr,"Memory Error: %s(%d) - Failed to allocate %lu bytes on stack.\n",
          __FILE__,__LINE__,sizeof_shapes_bytes(a));
  exit(1);  
}

/*
 * ZORDER
 */
#undef  sizeof_strides_bytes
#undef  sizeof_shapes_bytes
#define sizeof_strides_bytes(array) (sizeof(size_t)*((array)->ndims+1))
#define sizeof_shapes_bytes(array)  (sizeof(size_t)*((array)->ndims))

Array* ZOrder_Transform(Array *a)
{ size_t *strides, *shape;
  Array *out;
  out = Make_Array_With_Shape(a->kind,a->type,Array_Shape(a));
  if(!( strides = alloca(sizeof_strides_bytes(a)) )) goto MemoryError1;
  if(!( shape   = alloca(sizeof_shapes_bytes(a) ) )) goto MemoryError2;
  compute_strides_size_t(a,strides);
  copy_dims_size_t(a->ndims, a->dims, shape);
  zorder(a->ndims,shape,out->data,strides,a->data,strides);
  return out;
MemoryError1:
  fprintf(stderr,"Memory Error: %s(%d) - Failed to allocate %lu bytes on stack.\n",
      __FILE__,__LINE__,sizeof_strides_bytes(a));
  exit(1);
  
MemoryError2:
  fprintf(stderr,"Memory Error: %s(%d) - Failed to allocate %lu bytes on stack.\n",
          __FILE__,__LINE__,sizeof_shapes_bytes(a));
  exit(1);  
}

Array* ZOrder_Inverse_Transform(Array *a)
{ size_t *strides, *shape;
  Array *out;
  out = Make_Array_With_Shape(a->kind,a->type,Array_Shape(a));
  if(!( strides = alloca(sizeof_strides_bytes(a)) )) goto MemoryError1;
  if(!( shape   = alloca(sizeof_shapes_bytes(a) ) )) goto MemoryError2;
  compute_strides_size_t(a,strides);
  copy_dims_size_t(a->ndims, a->dims, shape);
  izorder(a->ndims,shape,out->data,strides,a->data,strides);
  return out;
MemoryError1:
  fprintf(stderr,"Memory Error: %s(%d) - Failed to allocate %lu bytes on stack.\n",
      __FILE__,__LINE__,sizeof_strides_bytes(a));
  exit(1);
  
MemoryError2:
  fprintf(stderr,"Memory Error: %s(%d) - Failed to allocate %lu bytes on stack.\n",
          __FILE__,__LINE__,sizeof_shapes_bytes(a));
  exit(1);  
}
