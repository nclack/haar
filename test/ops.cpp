#include <internal/ops.h>
#include <gtest/gtest.h>
#include <array.h>
#include "helpers.h"

typedef float f32;
#define TOL_F32 1e-5

// DATA

static const f32 data01[] = {
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
}; // 5 x 3 x 4

static const f32 data02[] = {
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
  
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
  
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
  
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
}; // 5 x 3 x 4

static const f32 expt01[] = {
  1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
  1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
  1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
  
  1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
  1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
  1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
  
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
}; // 5 x 3 x 4

static const f32 expt02[] = {
  1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
  1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
  1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
  
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  
  1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
  1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
  1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
  
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
}; // 5 x 3 x 4

static const f32 expt03[] = {
  2.0f, 2.0f, 0.0f, 0.0f, 0.0f,
  2.0f, 2.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  
  1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
  1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  
  1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
  1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
}; // 5 x 3 x 4

// TESTS

class Ops3DF32:public ::testing::Test
{ public:
  size_t shape[3],strides[4];
  f32 *zeros,
      *ones,
      *e1,
      *e2,
      *e3,
      *result;

  virtual void SetUp()
  { size_t length = sizeof(data01)/sizeof(f32);
    zeros  = new f32[length];
    ones   = new f32[length];
    e1     = new f32[length];
    e2     = new f32[length];
    e3     = new f32[length];
    result = new f32[length];
    memcpy(zeros,data01,sizeof(data01));
    memcpy(ones,data02,sizeof(data02));
    memcpy(e1,expt01,sizeof(expt01));
    memcpy(e2,expt02,sizeof(expt02));
    memcpy(e3,expt03,sizeof(expt03));
    shape[0] = 5;
    shape[1] = 3;
    shape[2] = 4;
    size_t i;
    strides[0] = 1;
    for(i=0;i<3;++i)
      strides[i+1] = strides[i]*shape[i]; //1,5,15,60
  }

  virtual void TearDown()
  { delete [] zeros;
    delete [] e1;
    delete [] e2;    
    delete [] e3;
    delete [] result;
  }
};

TEST_F(Ops3DF32,Copy)
{ size_t sh[]  = {3,3,2};
  copy(3,sh,zeros,strides,ones,strides);
  ASSERT_NEAR(0.0f,RMSE<f32>(strides[3],zeros,e1),TOL_F32);
}

TEST_F(Ops3DF32, Swap)
{ size_t sh[]  = {3,3,2};
  copy(3,sh,zeros,strides,ones,strides);
  sh[0]=5;
  sh[2]=1;
  swap(3,sh,zeros+strides[2],strides,zeros+2*strides[2],strides);
  ASSERT_NEAR(0.0f,RMSE<f32>(strides[3],zeros,e2),TOL_F32);
}

TEST_F(Ops3DF32, Gather)
{ gather(2, 3, shape, e2, strides);
  ASSERT_NEAR(0.0, RMSE<f32>(strides[3],e1,e2), TOL_F32);
}

TEST_F(Ops3DF32, Scatter)
{ scatter(2, 3, shape, e1, strides);
  ASSERT_NEAR(0.0, RMSE<f32>(strides[3],e1,e2), TOL_F32);
}

TEST_F(Ops3DF32, Add)
{ size_t sh[] = {2,2,3};
  copy(3,shape,result,strides,zeros,strides);
  add(3,sh,result,strides,e1,strides,e2,strides);
  ASSERT_NEAR(0.0, RMSE<f32>(strides[3],result,e3), TOL_F32);
}