# Synthetic colour study

`synthetic.ppm` is a deterministic 256×256 RGB test image created for this project,
released under CC0-1.0. It is not a photograph or an ImageNet example. No external
image or uncertain licence is used. There is no source URL: the source is the
formula below. File size: 196,623 bytes (under 200 KB).

At integer (x,y), RGB = (x, y, 240 if (floor(x/24)+floor(y/24)) is odd, otherwise 35).
PPM P6, max value 255. The producer supports P6 RGB samples, resizes the short side
to 256 with nearest-neighbour sampling and takes a centre 224×224 crop. It subtracts
RGB constants (123,117,104), without dividing by 255. This documented placeholder
is **not the ANX-12 pipeline**. The replay displays the actual cropped input.
