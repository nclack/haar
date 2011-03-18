#include <array.h>

int main()
{ Dimn_Type d[] = {3,4,5};
  Array *a = Make_Array(PLAIN_KIND,UINT8_TYPE,3,d);
  printf("Address: 0x%p\n");
  return 0;
}
