#include <array.h>
#include <image.h>
#include <utilities.h>
#include <haar-mylib.h>
#include <fft.h>       //for Padding routines

char *spec[] = {"<out:TIFF> <in:TIFF>",0};

void pad(Array *a)
{
  Coordinate *s=Array_Shape(a);
  Dimn_Type i,*d;
  d = (Dimn_Type*) s->data;
  for(i=0;i<s->size;++i)
    d[i] = Power_Of_2_Pad(d[i]);
  Pad_Array_Inplace(a, Idx2CoordA(a, 0), s);
}

int main(int argc, char *argv[])
{ Process_Arguments(argc, argv, spec, 1);
  { Array *a;
    a = Read_Image(Get_String_Arg("in"), 0);
    Convert_Array_Inplace(a, PLAIN_KIND, FLOAT32_TYPE, 32, 1.0);
    pad(a);
    Haar_Transform(a);
    Write_Image(Get_String_Arg("out"), a, 0);
    Free_Array(a);
  }
}
