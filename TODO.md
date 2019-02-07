# TODOs
- [x] Update setup.py
- [x] General code cleaning and commenting
- Update manual[pending]
- Move sampleNodes inside to reduce load on the memory. This prolly
requires to rewrite some work to lose the np dependency
- [x] The rng generation is currently precomputed; this will cause issues for larger graphs;
 preferable this would occur closer to the source or thread safe manner (see prev point)
- Find out how to properly wrap unordered_map with vector as keys or use concurrent variants. This will remove encoding dependency

# Mind scribbles
- [x] Sampler is currently using rand from stdlib. This is bad.
However the Mersenne twisters require the lockdown of the GIL
which is annoying. According to the web mersenne twister is thread safe > don't get what I am doing wrong but the annotations
indicate python interactions [done; replaced by mersenne]
- [x] The class currently isn't fully picklable. This is due to the memory views.
The memory views make it very fast (c array fast). Changing the defintions in pxd to np.darray fixes
this issues, however it reduces the speed somewhat. The ultimate goal would be to replace it with full
c arrays instead of memviews. I did try it with cpp vectors but it was (for some reason) slower.


# Notes on threading
- [x] UPDATE: threading currently works, but notes below may still be useful for future me
For some reasons using prange has some problems with it. First,  it requires
separate models to be accessed independently. Although this should be possible
and I attempted to do this on rewrite-vectors branch, it produces erroneous results. Performing everything single thread/core is the fastest currently.
Ideally I would want to utilize all the cores to compute the snapshots repeats
in parallel.


- [x] Found the segfault; writing to the buffer is not threadsafe, even though the keys are unique. (wasn't it was a wrongful idx)
- [x] If you replace the memviews with ndarrays and slice into them, the speed will be reduces by 300 percent. This is the cause of issues in mp
