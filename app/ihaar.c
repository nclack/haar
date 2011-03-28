#include <array.h>
#include <image.h>
#include <utilities.h>
#include <haar-mylib.h>


char *spec[] = {"<out:TIFF> <in:TIFF>",0};

int main(int argc, char *argv[])
{ Process_Arguments(argc, argv, spec, 1);
  { Array *a;
    a = Read_Image(Get_String_Arg("in"), 0);
    Convert_Array_Inplace(a, PLAIN_KIND, FLOAT32_TYPE, 32);
    Haar_Inverse_Transform(a);
    Write_Image(Get_String_Arg("out"), a, 0);
    Free_Array(a);
  }
}