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

static const f32 data02[] = {
  1.0f,  2.0f,  3.0f   , 4.0f,
  5.0f,  6.0f,  7.0f   , 8.0f,
  9.0f, 10.0f, 11.0f   ,12.0f,
 13.0f, 14.0f, 15.0f   ,16.0f,

 17.0f, 18.0f, 19.0f   ,20.0f,
 21.0f, 22.0f, 23.0f   ,24.0f,
 25.0f, 26.0f, 27.0f   ,28.0f,
 29.0f, 30.0f, 31.0f   ,32.0f,

 33.0f, 34.0f, 35.0f   ,36.0f,
 37.0f, 38.0f, 39.0f   ,40.0f,
 41.0f, 42.0f, 43.0f   ,44.0f,
 45.0f, 46.0f, 47.0f   ,48.0f,

 49.0f, 50.0f, 51.0f   ,52.0f,                           
 53.0f, 54.0f, 55.0f   ,56.0f,
 57.0f, 58.0f, 59.0f   ,60.0f,
 61.0f, 62.0f, 63.0f   ,64.0f,
};


static const f32 expt02[] = {
  1.0f,  2.0f,  5.0f   , 6.0f,
 17.0f, 18.0f, 21.0f   ,22.0f,

  3.0f,  4.0f,  7.0f   , 8.0f,
 19.0f, 20.0f, 23.0f   ,24.0f,

  9.0f, 10.0f, 13.0f   ,14.0f,
 25.0f, 26.0f, 29.0f   ,30.0f,

 11.0f, 12.0f, 15.0f   ,16.0f,
 27.0f, 28.0f, 31.0f   ,32.0f,

 33.0f, 34.0f, 37.0f   ,38.0f,
 49.0f, 50.0f, 53.0f   ,54.0f,

 35.0f, 36.0f, 39.0f   ,40.0f,
 51.0f, 52.0f, 55.0f   ,56.0f,

 41.0f, 42.0f, 45.0f   ,46.0f,
 57.0f, 58.0f, 61.0f   ,62.0f,

 43.0f, 44.0f, 47.0f   ,48.0f,
 59.0f, 60.0f, 63.0f   ,64.0f,
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

/*******************************************************************************
 * 3D f32
 ******************************************************************************/
class ZOrder3DF32:public ::testing::Test
{
  public:
    size_t length,istride[4],ostride[4],shape[3];
    f32 *data;
    f32 *expt;
    f32 *res;

    virtual void SetUp() 
    { length = sizeof(data02)/sizeof(f32);
      data = new f32[length];
      expt = new f32[length];
      res  = new f32[length];
      data = (f32*)memcpy(data,data02,sizeof(data02));
      expt = (f32*)memcpy(expt,expt02,sizeof(expt02));
      istride[0] = 1;
      istride[1] = 4;
      istride[2] = 4*istride[1];
      istride[3] = length;
      shape[0] = istride[1]/istride[0];
      shape[1] = istride[2]/istride[1];
      shape[2] = istride[3]/istride[2];
      memcpy(ostride,istride,sizeof(ostride));
    }
    virtual void TearDown() 
    { delete [] data;
      delete [] expt;
      delete [] res;
    }
};

TEST_F(ZOrder3DF32,Forward)
{
  zorder(3,shape,res,ostride,data,istride);
  ASSERT_NEAR(0.0f,RMSE<f32>(length,res,expt),TOL_F32);
}

TEST_F(ZOrder3DF32,SelfInverse)
{
  zorder(3,shape,res,ostride,data,istride);
  zorder(3,shape,expt,ostride,res,istride);
  ASSERT_NEAR(0.0f,RMSE<f32>(length,data,expt),TOL_F32);
}
