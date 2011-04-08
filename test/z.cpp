#include <z.h>
#include <gtest/gtest.h>
#include "helpers.h"

typedef float f32;
static const f32 data01[] = {
  1.0f,  2.0f,  3.0f   , 4.0f,
  5.0f,  6.0f,  7.0f   , 8.0f,
  9.0f, 10.0f, 11.0f   ,12.0f,
 13.0f, 14.0f, 15.0f   ,16.0f,
};
static const f32 expt01[] = {
  1.0f,  2.0f,  5.0f   , 6.0f,
  3.0f,  4.0f,  7.0f   , 8.0f,
  9.0f, 10.0f, 13.0f   ,14.0f,
 11.0f, 12.0f, 15.0f   ,16.0f,
};

/*******************************************************************************
 * 2D f32
 ******************************************************************************/
class ZOrder2DF32:public ::testing::Test
{
  public:
    size_t length,istride[3],ostride[3],shape[2];
    f32 *data;
    f32 *expt;
    f32 *res;

    virtual void SetUp() 
    { length = sizeof(data01)/sizeof(f32);
      data = new f32[length];
      expt = new f32[length];
      res  = new f32[length];
      data = (f32*)memcpy(data,data01,sizeof(data01));
      expt = (f32*)memcpy(expt,expt01,sizeof(expt01));
      istride[0] = 1;
      istride[1] = 4;
      istride[2] = length;
      shape[0] = istride[1]/istride[0];
      shape[1] = istride[2]/istride[1];
      memcpy(ostride,istride,sizeof(ostride));
    }
    virtual void TearDown() 
    { delete [] data;
      delete [] expt;
      delete [] res;
    }
};

TEST_F(ZOrder2DF32,Forward)
{
  zorder(2,shape,res,ostride,data,istride);
  ASSERT_NEAR(0.0f,RMSE<f32>(length,res,expt),TOL_F32);
}

TEST_F(ZOrder2DF32,SelfInverse)
{
  zorder(2,shape,res,ostride,data,istride);
  zorder(2,shape,expt,ostride,res,istride);
  ASSERT_NEAR(0.0f,RMSE<f32>(length,data,expt),TOL_F32);
}
