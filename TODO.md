- Sampler is currently using rand from stdlib. This is bad.
However the Mersenne twisters require the lockdown of the GIL
which is annoying. According to the web mersenne twister is thread safe > don't get what I am doing wrong but the annotations
indicate pythnon interactions.
