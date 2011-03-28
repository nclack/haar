#include <haar.h>
#include <gtest/gtest.h>
#include "helpers.h"

typedef float f32;
static const f32 data01[] = {1.0f,  2.0f,  3.0f   , 4.0f    };
static const f32 expt01[] = {5.0f, -2.0f, -0.7071f, -0.7071f};
static const f32 data02[] = { 1.0f,  2.0f,  3.0f   , 4.0f,
                              5.0f,  6.0f,  7.0f   , 8.0f,  
                              9.0f, 10.0f, 11.0f   ,12.0f,  
                             13.0f, 14.0f, 15.0f   ,16.0f};
static const f32 expt02[] = {34.0f, -4.0f, -1.0f   ,-1.0f,
                            -16.0f,  0.0f, -1.0f   ,-1.0f,  
                             -4.0f, -4.0f,  0.0f   , 0.0f,  
                             -4.0f, -4.0f,  0.0f   , 0.0f};

#define TOL_F32 1e-5

/*******************************************************************************
 * 1D f32
 ******************************************************************************/
class Haar1DF32:public ::testing::Test
{
  public:
    size_t length,istride[2],ostride[2];
    f32 *data;
    f32 *expt;
    f32 *res;
    HaarWorkspace ws;

    virtual void SetUp() 
    { length = sizeof(data01)/sizeof(f32);
      data = new f32[length];
      expt = new f32[length];
      res  = new f32[length];
      data = (f32*)memcpy(data,data01,sizeof(data01));
      expt = (f32*)memcpy(expt,expt01,sizeof(expt01));
      istride[0] = 1;
      istride[1] = length;
      ostride[0] = 1;
      ostride[1] = length;
      HaarWorkspaceInit(&ws);
    }
    virtual void TearDown() 
    { delete [] data;
      delete [] expt;
      delete [] res;
      HaarWorkspaceClean(&ws);
    }
};

TEST_F(Haar1DF32,Forward)
{ 
  haar(&ws,1,&length,res,ostride,data,istride);
  ASSERT_NEAR(0.0f,RMSE<f32>(length,res,expt),TOL_F32);
}

TEST_F(Haar1DF32,ForwardInPlace)
{
  haar(&ws,1,&length,data,istride,data,istride);
  ASSERT_NEAR(0.0f,RMSE<f32>(length,data,expt),TOL_F32);
}

TEST_F(Haar1DF32,Inverse)
{ 
  ihaar(&ws,1,&length,res,ostride,expt,istride);
  ASSERT_NEAR(0.0f,RMSE<f32>(length,res,data),TOL_F32);
}

TEST_F(Haar1DF32,InverseInPlace)
{ 
  ihaar(&ws,1,&length,expt,istride,expt,istride);
  ASSERT_NEAR(0.0f,RMSE<f32>(length,expt,data),TOL_F32);
}

/*******************************************************************************
 * 2D f32
 ******************************************************************************/
class Haar2DF32:public ::testing::Test
{
  public:
    size_t length,istride[3],ostride[3],shape[2];
    f32 *data;
    f32 *expt;
    f32 *res;
    HaarWorkspace ws;

    virtual void SetUp() 
    { length = sizeof(data02)/sizeof(f32);
      data = new f32[length];
      expt = new f32[length];
      res  = new f32[length];
      data = (f32*)memcpy(data,data02,sizeof(data02));
      expt = (f32*)memcpy(expt,expt02,sizeof(expt02));
      istride[0] = 1;
      istride[1] = 4;
      istride[2] = length;
      shape[0] = istride[1]/istride[0];
      shape[1] = istride[2]/istride[1];
      memcpy(ostride,istride,sizeof(ostride));
      HaarWorkspaceInit(&ws);
    }
    virtual void TearDown() 
    { delete [] data;
      delete [] expt;
      delete [] res;
      HaarWorkspaceClean(&ws);
    }
};

TEST_F(Haar2DF32,Forward)
{ 
  haar(&ws,2,shape,res,ostride,data,istride);
  ASSERT_NEAR(0.0f,RMSE<f32>(length,res,expt),TOL_F32);
}

TEST_F(Haar2DF32,ForwardInPlace)
{
  haar(&ws,2,shape,data,istride,data,istride);
  ASSERT_NEAR(0.0f,RMSE<f32>(length,data,expt),TOL_F32);
}

TEST_F(Haar2DF32,Inverse)
{ 
  ihaar(&ws,2,shape,res,ostride,expt,istride);
  ASSERT_NEAR(0.0f,RMSE<f32>(length,res,data),TOL_F32);
}

TEST_F(Haar2DF32,InverseInPlace)
{ 
  ihaar(&ws,2,shape,expt,istride,expt,istride);
  ASSERT_NEAR(0.0f,RMSE<f32>(length,expt,data),TOL_F32);
}
