# TODOs
- Update setup.py [done]
- General code cleaning and commenting [partial]
- Update manual



# Mind scribbles
- Sampler is currently using rand from stdlib. This is bad.
However the Mersenne twisters require the lockdown of the GIL
which is annoying. According to the web mersenne twister is thread safe > don't get what I am doing wrong but the annotations
indicate pythnon interactions [done; replaced by mersenne]
- The class currently isn't fully picklable. This is due to the memory views.
The memory views make it very fast (c array fast). Changing the defintions in pxd to np.darray fixes
this issues, however it reduces the speed somewhat. The ultimate goal would be to replace it with full
c arrays instead of memviews. I did try it with cpp vectors but it was (for some reason) slower.


# Notes on threading
UPDATE: threading currently works, but notes below may still be useful for future me

For some reasons using prange has some problems with it. First,  it requires
separate models to be accessed independently. Although this should be possible
and I attempted to do this on rewrite-vectors branch, it produces erroneous results. Performing everything single thread/core is the fastest currently.
Ideally I would want to utilize all the cores to compute the snapshots repeats
in parallel.


- Found the segfault; writing to the buffer is not threadsafe, even though the keys are unique.
- If you replace the memviews with ndarrays and slice into them, the speed will be reduces by 300 percent. This is the cause of issues in mp
