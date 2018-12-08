# Mind scribbles
- Sampler is currently using rand from stdlib. This is bad.
However the Mersenne twisters require the lockdown of the GIL
which is annoying. According to the web mersenne twister is thread safe > don't get what I am doing wrong but the annotations
indicate pythnon interactions [done; replaced by mersenne]
- The class currently isn't fully picklable. This is due to the memory views.
The memory views make it very fast (c array fast). Changing the defintions in pxd to np.darray fixes
this issues, however it reduces the speed somewhat. The ultimate goal would be to replace it with full
c arrays instead of memviews. I did try it with cpp vectors but it was (for some reason) slower.
