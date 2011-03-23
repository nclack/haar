#include <haar.h>
#include <gtest/gtest.h>

float data01[] = {1.0f,  2.0f,  3.0f   , 4.0f    };
float expt01[] = {5.0f, -2.0f, -0.7071f, -0.7071f};

template<class T>
T meanSquaredErr(size_t n, T* a, T* b)
{ T ssq=0.0,t;
  for(size_t i=0;i<n;++i)
  { t = b[i]-a[i];
    ssq+=t*t;
  }
  return ssq/n;
}

TEST(Haar,Forward1DF32)
{ size_t n         = sizeof(data01)/sizeof(float),
         istride[] = {1,sizeof(data01)/sizeof(float)},
         ostride[] = {1,sizeof(data01)/sizeof(float)};
  float  result[sizeof(data01)/sizeof(float)];
  HaarWorkspace ws = HAAR_WORKSPACE_INIT;
  haar(&ws,1,&n,result,ostride,data01,istride);
  ASSERT_FLOAT_EQ(0.0f,meanSquaredErr<float>(n,result,expt01));
  HaarWorkspaceClean(&ws);
}

TEST(Haar,Inverse1DF32)
{ size_t n        = sizeof(data01)/sizeof(float),
         istride[] = {1,sizeof(data01)/sizeof(float)},
         ostride[] = {1,sizeof(data01)/sizeof(float)};
  float  result[sizeof(data01)/sizeof(float)];
  HaarWorkspace ws = HAAR_WORKSPACE_INIT;
  ihaar(&ws,1,&n,result,ostride,expt01,istride);
  ASSERT_FLOAT_EQ(0.0f,meanSquaredErr<float>(n,result,data01));
  HaarWorkspaceClean(&ws);
}
