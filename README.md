# Extract-Facial-Regions
Extracting Facial Regions using Dlib's facial landmark detection

The facial landmark detector implemented inside dlib produces 68 (x, y)-coordinates that map to specific facial structures. These 68 point mappings were obtained by training a shape predictor on the labeled iBUG 300-W dataset.

Examining the image, we can see that facial regions can be accessed via simple Python indexing (assuming zero-indexing with Python since the image above is one-indexed):

* Mouth - [48, 68].
* Right eyebrow - [17, 22].
* Left eyebrow - [22, 27].
* Right eye - [36, 42].
* Left eye - [42, 48].
* Nose - [27, 35].
* Jaw - [0, 17].

## Dependencies ##

* OpenCV
* Dlib
* Numpy






