typedef void HaarWorkspace;
void HaarWorkspaceClean(HaarWorkspace* ws);

void haar(HaarWorkspace* ws,
          size_t ndim, size_t* shape,
          float* out, size_t* ostrides,
          float* in, size_t* istrides);

void ihaar(HaarWorkspace* ws,
           size_t ndim, size_t* shape,
           float* out, size_t* ostrides,
           float* in, size_t* istrides);
