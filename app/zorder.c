#include <array.h>
#include <image.h>
#include <utilities.h>
#include <util-mylib.h>
#include <fft.h>       //for Padding routines

char *spec[] = {"[-1|--inverse] <out:TIFF> <in:TIFF>",0};

void pad(Array *a)
{
  Coordinate *s=Array_Shape(a);
  Dimn_Type i,*d,mx=0;
  d = (Dimn_Type*) s->data;
  for(i=0;i<s->size;++i)
  { Dimn_Type t;
    t = Power_Of_2_Pad(d[i]);
    mx = (t>mx)?t:mx;
  }
  for(i=0;i<s->size;++i)
    d[i] = mx;
  Pad_Array_Inplace(a, Idx2CoordA(a, 0), s);
}

int main(int argc, char *argv[])
{ Process_Arguments(argc, argv, spec, 1);
  { Array *a,*b;
    a = Read_Image(Get_String_Arg("in"), 0);
    Convert_Array_Inplace(a, PLAIN_KIND, FLOAT32_TYPE, 32, 1.0);
    pad(a);
    if(Is_Arg_Matched("-1")||Is_Arg_Matched("--inverse"))
      b = ZOrder_Inverse_Transform(a);
    else
      b = ZOrder_Transform(a);
    Free_Array(a);
    Write_Image(Get_String_Arg("out"), b, 0);
    Free_Array(b);
  }
}
