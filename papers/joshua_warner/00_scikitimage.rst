:author: Stéfan van der Walt
:email: stefanv@berkeley.edu
:institution: Berkeley Institute for Data Science, University of California at Berkeley, USA
:corresponding:

:author: Emmanuelle Gouillart
:email: emmanuelle.gouillart@nsup.org
:institution: Joint Unit CNRS/Saint-Gobain Surface of Glass and Interfaces, Aubervilliers, France
:equal-contributor:

:author: Alexandre F. de Siqueira
:email: alexandredesiqueira@programandociencia.com
:institution: University of Campinas, Campinas, Brazil
:institution: TU Bergakademie Freiberg, Freiberg, Germany
:equal-contributor:

:author: Joshua D. Warner
:email: joshua.dale.warner@gmail.com
:institution: Mayo Clinic, Rochester, USA
:equal-contributor:

------------
scikit-image
------------

.. class:: abstract

   scikit-image is an image processing library that implements algorithms and utilities for use in research, education and industry applications. It is released under the liberal Modified BSD open source license, provides a well-documented API in the Python programming language, and is developed by an active, international team of collaborators. In this chapter we highlight the advantages of open source to achieve the goals of  the scikit-image library, and we showcase several real-world image processing applications that use scikit-image. More information can be found on the project homepage, http://scikit-image.org.

.. class:: keywords

   image processing, computer vision, python

Introduction
------------

scikit-image is an image processing…

Mention how ndarray allows us to fit in with rest of eco-system

Parallel & distributed processing via dask


Panorama Stitching
------------------

This example stitches three images into a seamless panorama using several tools in scikit-image, including feature detection [Rub11]_, RANdom SAmple Consensus (RANSAC) [Fis81]_, graph theory, and affine transformations.  The images used in this example are available at https://github.com/scikit-image/skimage-tutorials/tree/master/images/pano named ``JDW_9*.jpg``, released under the CC-BY 4.0 by the author.

Load images
***********

The ``io`` module in scikit-image allows images to be loaded and saved. In this case the color panorama images will be loaded into an iterable `ImageCollection`, though one could also load them individually.

.. code-block:: python

   from skimage import io
   pano_images = io.ImageCollection(
       '/path/to/images/JDW_9*')

.. figure:: pano0_originals.png
   :align: center
   :figclass: w
   :scale: 60%

   Panorama source images, taken on the trail to Delicate Arch in Arches National Park, USA.  Released under CC-BY 4.0 by Joshua D. Warner. :label:`fig-pano0`

Feature detection and matching
******************************

To correctly align the images, a *projective* transformation relating them is required.

1. Define one image as a *target* or *destination* image, which will remain anchored while the others are warped.
2. Detect features in all three images.
3. Match features from left and right images against the features in the center, anchored image.

In this series, the middle image is the logical anchor point.  Numerous feature detection algorithms are available; this example will use Oriented FAST and rotated BRIEF (ORB) features available as ``skimage.feature.ORB`` [Rub11]_.

.. code-block:: python

   import matplotlib.pyplot as plt
   from skimage.color import rgb2gray
   from skimage.feature import (ORB, match_descriptors,
                                plot_matches)

   # Initialize ORB
   orb = ORB(n_keypoints=800, fast_threshold=0.05)
   keypoints = []
   descriptors = []

   # Detect features
   for image in pano_images:
       orb.detect_and_extract(rgb2gray(image))
       keypoints.append(orb.keypoints)
       descriptors.append(orb.descriptors)

   # Match features from images 0 -> 1 and 2 -> 1
   matches01 = match_descriptors(descriptors[0],
                                 descriptors[1],
                                 cross_check=True)
   matches12 = match_descriptors(descriptors[1],
                                 descriptors[2],
                                 cross_check=True)

   # Show raw matched features from left to center
   fig, ax = plt.subplots()
   plot_matches(ax, pano_images[0], pano_images[1],
                keypoints[0], keypoints[1], matches01)
   ax.axis('off');

.. figure:: pano1_ORB-raw.png
   :align: center

   Matched ORB keypoints from left and center images from :ref:`fig-pano0`. Most features line up similarly, but there are a number of obvious outliers or false matches. :label:`fig-pano1`

Transform estimation
********************

To filter out the false matches observed in Figure :ref:`fig-pano1`, RANdom SAmple Consensus (RANSAC) is used [Fis81]_. RANSAC is a powerful method of rejecting outliers available in ``skimage.transform.ransac``. The transformation is estimated using an iterative process based on randomly chosen subsets, finally selecting the model which corresponds best with the majority of matches.

It is important to note the randomness inherent to RANSAC. The results are robust, but will vary slightly every time.  Thus, it is expected that readers' results will deviate slightly from the published figures after this point.

.. code-block:: python

   from skimage.measure import ransac
   from skimage.transform import ProjectiveTransform

   # Keypoints from left (src) to middle (dst) images
   src = keypoints0[matches01[:, 0]][:, ::-1]
   dst = keypoints1[matches01[:, 1]][:, ::-1]

   model_ransac01, inliers01 = ransac(
       (src, dst), ProjectiveTransform, min_samples=4,
       residual_threshold=1, max_trials=300)

   # Keypoints from right (src) to middle (dst) images
   src = keypoints2[matches12[:, 1]][:, ::-1]
   dst = keypoints1[matches12[:, 0]][:, ::-1]

   model_ransac12, inliers12 = ransac(
       (src, dst), ProjectiveTransform, min_samples=4,
       residual_threshold=1, max_trials=300)

   # Show robust, RANSAC-matched features
   fig, ax = plt.subplots()
   plot_matches(ax, pano_images[0], pano_images[1],
                keypoints[0], keypoints[1],
                matches01[inliers01])
   ax.axis('off');

The results of robust transform estimation with RANSAC are shown in Figure :ref:`fig-pano2`.

.. figure:: pano2_ORB-RANSAC.png
   :align: center

   The best RANSAC transform estimation uses only these keypoints. The outliers are now excluded (compare with Figure :ref:`fig-pano1`). :label:`fig-pano2`

Warp images into place
**********************

Find appropriate canvas size
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before producing the panorama, the correct size for a new canvas to hold all three warped images is needed.  The entire size, or extent, of this image is carefully found.

.. code-block:: python

   from skimage.transform import SimilarityTransform

   # All three images have the same size
   r, c = pano_images[1].shape[:2]

   # Note that transformations take coordinates in
   # (x, y) format, not (row, column), for literature
   # consistency
   corners = np.array([[0, 0],
                       [0, r],
                       [c, 0],
                       [c, r]])

   # Warp image corners to their new positions
   warped_corners01 = model_ransac01(corners)
   warped_corners12 = model_ransac12(corners)

   # Extents of both target and warped images
   all_corners = np.vstack((warped_corners01,
                            warped_corners12,
                            corners))

   # Overall output shape is max - min
   corner_min = np.min(all_corners, axis=0)
   corner_max = np.max(all_corners, axis=0)
   output_shape = (corner_max - corner_min)

   # Ensure integer shape
   output_shape = np.ceil(
       output_shape[::-1]).astype(int)


Next, each image is warped and placed into a new canvas of shape ``output_shape``.

Translate middle target image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The middle image is stationary, but still needs to be shifted into the center of the larger canvas.  This is done with simple translation using a ``SimilarityTransform``.

.. code-block:: python

   from skimage.transform import warp, SimilarityTransform

   offset1 = SimilarityTransform(translation= -corner_min)

   # Translate pano1 into place
   pano1_warped = warp(
       pano_images[1], offset1.inverse, order=3,
       output_shape=output_shape, cval=-1)

   # Acquire the image mask for later use
   # Mask == 1 inside image, then return backgroun to 0
   pano1_mask = (pano1_warped != -1)[..., 0]
   pano1_warped[~pano1_mask] = 0


Apply RANSAC-estimated transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The other two images are warped by ``ProjectiveTransform`` into place.

.. code-block:: python

   # Warp left image
   transform01 = (model_ransac01 + offset1).inverse
   pano0_warped = warp(
       pano_images[0], transform01, order=3,
       output_shape=output_shape, cval=-1)

   # Mask == 1 inside image, then return backgroun to 0
   pano0_mask = (pano0_warped != -1)[..., 0]
   pano0_warped[~pano0_mask] = 0

   # Warp right image
   transform12 = (model_ransac12 + offset1).inverse
   pano2_warped = warp(
       pano_images[2], transform12, order=3,
       output_shape=output_shape, cval=-1)

   # Mask == 1 inside image, then return backgroun to 0
   pano2_mask = (pano2_warped != -1)[..., 0]
   pano1_warped[~pano1_mask] = 0

See the warped images in :ref:`fig-pano3`.

.. figure:: pano3_warped.png
   :align: center

   Each image is now correctly warped into a new frame with room for the others, ready to be composited/stitched together. :label:`fig-pano3`


Image stitching using minimum-cost path
***************************************

Because of optical non-linearities, simply averaging these images together will not work. The overlapping areas become significantly blurred.  Instead, a minimum-cost path can be found with the assistance of ``skimage.graph.route_through_array``. This function allows one to

* start at any point on an array
* find a particular path to any other point in the array
* the path found *minimizes* the sum of values on the path.

The array in this instance is a *cost array* which is carefully defined so the path found will be desired one, while the path itself is the *minimum-cost path*, or MCP. To use this technique we need starting and ending points, as well as a cost array.

Define seed points
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   ymax = output_shape[1] - 1
   xmax = output_shape[0] - 1

   # Start anywhere along the top and bottom
   mask_pts01 = [[0,    ymax // 3],
                 [xmax, ymax // 3]]

   # Start anywhere along the top and bottom
   mask_pts12 = [[0,    2*ymax // 3],
                 [xmax, 2*ymax // 3]]


Construct cost array
^^^^^^^^^^^^^^^^^^^^
:label:`construct-costs`

For optimal results, great care goes into the creation of the cost array.  The function below is designed to construct the best possible cost array.  Its tasks are:

1. Start with a high-cost image filled with ones.
2. Use the mask - which defines where the overlapping region will be - to find the distance from the top/bottom edges to the masked area.
3. Reject mostly vertical areas.
4. Give a cost break to areas slightly further away, if the warped overlap is not parallel with the image edges, to ensure fair competition
5. Put the absolute value of the *difference* of the overlapping images in place

A convenience function ``generate_costs`` is provided in the Appendix which accomplishes the above.

.. code-block:: python

  # Use the generate_costs function
  costs01 = generate_costs(pano0_warped - pano1_warped,
                           pano0_mask & pano1_mask)
  costs12 = generate_costs(pano1_warped - pano2_warped,
                           pano1_mask & pano2_mask)


Find minimum-cost path and masks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the cost function is generated, the minimum cost path can be found simply and efficiently.

.. code-block:: python

   from skimage.graph import route_through_array

   # Find the MCP
   pts01, _ = route_through_array(
     costs01, mask_pts01[0], mask_pts01[1],
     fully_connected=True)

   pts01 = np.array(pts01)

   # Create final mask for the left image
   mask0 = np.zeros_like(pano0_warped[..., 0],
                         dtype=np.uint8)
   mask0[pts01[:, 0], pts01[:, 1]] = 1
   mask0 = (
     label(mask0, connectivity=1, background=-1) == 1)


.. figure:: pano4_mcp.png
   :align: center
   :figclass: w
   :scale: 98%

   The minimum cost path in blue is the ideal stitching boundary. It stays as close to zero (mid-gray) as possible throughout its path.  The background is the cost array, with zero set to mid-gray for better visibility.  Note the subtle shading effect of cost reduction below the difference region.  Readers' paths may differ in appearance, but are optimal for their RANSAC-chosen transforms.

Because ``mask0`` is a *final* mask for the left image, it needs to constrain the solution for the right image. This step is essential if there is large overlap such that the left and right images could theoretically occupy the same space.  It ensures the MCPs will not cross.

.. code-block:: python

   # New constraint modifying cost array
   costs12[mask0 > 0] = 1

   pts12, _ = route_through_array(
     costs12, mask_pts12[0], mask_pts12[1],
     fully_connected=True)

   pts12 = np.array(pts12)

   # Final mask for right image
   mask2 = np.zeros_like(mask0, dtype=np.uint8)
   mask2[pts12[:, 0], pts12[:, 1]] = 1
   mask2 = (
     label(mask2, connectivity=1, background=-1) == 3)

   # Mask for middle image is one of exclusion
   mask1 = ~(mask0 | mask2).astype(bool)


Blend images together with alpha channels
*****************************************

Most image formats can support an alpha channel as an optional fourth channel, which defines the transparency at each pixel.  We now have three warped images and three corresponding masks.  These masks can be incorporated as alpha channels to seamlessly blend them together.

.. code-block:: python

   # Convenience function for alpha blending
   def add_alpha(img, mask=None):
     """
     Adds a masked alpha channel to an image.

     Parameters
     ----------
     img : (M, N[, 3]) ndarray
         Image data, should be rank-2 or rank-3
         with RGB channels
     mask : (M, N[, 3]) ndarray, optional
         Mask to be applied. If None, the alpha channel
         is added with full opacity assumed (1) for all
         locations.
     """
     from skimage.color import gray2rgb
     if mask is None:
       mask = np.ones_like(img)

     if img.ndim == 2:
       img = gray2rgb(img)

     return np.dstack((img, mask))

   # Applying this function
   left_final = add_alpha(pano0_warped, mask0)
   middle_final = add_alpha(pano1_warped, mask1)
   right_final = add_alpha(pano2_warped, mask2)


Matplotlib's ``imshow`` supports alpha blending, but the default interpolation mode causes edge effects [Hunt07]_.  So as we create our final composite image, interpolation is disabled.

.. code-block:: python

   fig, ax = plt.subplots()

   # Turn off matplotlib's interpolation
   ax.imshow(left_final, interpolation='none')
   ax.imshow(middle_final, interpolation='none')
   ax.imshow(right_final, interpolation='none')

   ax.axis('off')
   fig.tight_layout()
   fig.show()

.. figure:: pano5_final.png
   :align: center
   :figclass: w
   :scale: 31%

   The final, seamlessly stitched panorama.

References
----------
.. [Hunt07] Hunter, J. D. *Matplotlib: A 2D graphics environment*,
            Computing In Science & Engineering, 9(3):90-95, 2007.
            DOI:10.5281/zenodo.61948

.. [Rub11] Rublee, E.; Rabaud, V.; Konolige, K.; Bradski, G.
           *ORB: an efficient alternative to SIFT or SURF*,
           IEEE International Conference on Computer Vision (ICCV),
           2564-2571, 2011. DOI:10.1109/ICCV.2011.6126544

.. [Fis81] Fischler, M. A.; Robert C. B. *Random sample consensus:
           a paradigm for model fitting with applications to image
           analysis and automated cartography.* Communications of
           the ACM, 24(6):381-395, 1981.


Appendix
--------

This supplemental appendix includes convenience functions which were deemed obstructive for the flow of the main chapter text.  They are referenced where appropriate above.  Including them resulted in more elegant and intuitive examples.

Minimum-cost-path cost array creation
*************************************
:label:`cost-arr-func`

This function generates an ideal cost array for panorama stitching, using the principles set forth in :ref:`construct-costs`.

.. code-block:: python

   from skimage.measure import label

   def generate_costs(diff_image, mask, vertical=True,
                      gradient_cutoff=2.,
                      zero_edges=True):
     """
     Ensure equal-cost paths from edges to
     region of interest.

     Parameters
     ----------
     diff_image : (M, N) ndarray of floats
         Difference of two overlapping images.
     mask : (M, N) ndarray of bools
         Mask representing the region of interest in
         ``diff_image``.
     vertical : bool
         Control if stitching line is vertical or
         horizontal.
     gradient_cutoff : float
         Controls how far out of parallel lines can
         be to edges before correction is terminated.
         The default (2.) is good for most cases.
     zero_edges : bool
         If True, the edges are set to zero so the
         seed is not bound to any specific horizontal
         location.

     Returns
     -------
     costs_arr : (M, N) ndarray of floats
         Adjusted costs array, ready for use.
     """
     if vertical is not True:  # run transposed
       return tweak_costs(
         diff_image.T, mask.T, vertical=True,
         gradient_cutoff=gradient_cutoff).T

     # Start with a high-cost array of 1's
     diff_image = rgb2gray(diff_image)
     costs_arr = np.ones_like(diff_image)

     # Obtain extent of overlap
     row, col = mask.nonzero()
     cmin = col.min()
     cmax = col.max()

     # Label discrete regions
     cslice = slice(cmin, cmax + 1)
     labels = label(mask[:, cslice], background=-1)

     # Find distance from edge to region
     upper = (labels == 1).sum(axis=0)
     lower = (labels == 3).sum(axis=0)

     # Reject areas of high change
     ugood = np.abs(
       np.gradient(upper)) < gradient_cutoff
     lgood = np.abs(
       np.gradient(lower)) < gradient_cutoff

     # Cost break to areas slightly farther from edge
     costs_upper = np.ones_like(upper,
                                dtype=np.float64)
     costs_lower = np.ones_like(lower,
                                dtype=np.float64)
     costs_upper[ugood] = (
         upper.min() / np.maximum(upper[ugood], 1))
     costs_lower[lgood] = (
         lower.min() / np.maximum(lower[lgood], 1))

     # Expand from 1d back to 2d
     vdis = mask.shape[0]
     costs_upper = (
       costs_upper[np.newaxis, :].repeat(vdis, axis=0))
     costs_lower = (
       costs_lower[np.newaxis, :].repeat(vdis, axis=0))

     # Place these in output array
     costs_arr[:, cslice] = costs_upper * (labels==1)
     costs_arr[:, cslice] += costs_lower * (labels==3)

     # Finally, place the difference image
     costs_arr[mask] = np.abs(diff_image[mask])

     if zero_edges is True:  # top & bottom rows = zero
       costs_arr[0, :] = 0
       costs_arr[-1, :] = 0

     return costs_arr


Flood fill
**********
:label:`flood-fill`

This Cython function is a basic flood fill algorithm which accepts an array and modifies it in place.  It starts at a defined point, which is changed to a new value, then iteratively fills outward by seeking all connected points which had the original value, changing them as well.

The conceptual analogy of this algorithm is the "bucket" tool in many photo editing programs.

.. code-block:: cython

   import numpy as np
   cimport numpy as cnp


   # Compiler directives
   @cython.cdivision(True)
   @cython.boundscheck(False)
   @cython.nonecheck(False)
   @cython.wraparound(False)
   def flood_fill(unsigned char[:, ::1] image,
                  tuple start_coords,
                  Py_ssize_t fill_value):
     """
     Flood fill algorithm

     Parameters
     ----------
     image : (M, N) ndarray of uint8 type
         Image with flood to be filled. Modified
         inplace.
     start_coords : tuple
         Length-2 tuple of ints defining (row, col)
         start coordinates.
     fill_value : int
         Value to fill flooded area with.

     Returns
     -------
     None. ``image`` is modified inplace.
     """
     cdef:
       Py_ssize_t x, y, xsize, ysize, orig_value
       set stack

     xsize = image.shape[0]
     ysize = image.shape[1]
     orig_value = image[start_coords[0],
                        start_coords[1]]

     if fill_value == orig_value:
       raise ValueError(
         "Filling region with same value "
         "already present is unsupported. "
         "Did you already fill this region?")

     stack = set(((start_coords[0],
                   start_coords[1]), ))

     while stack:
       x, y = stack.pop()

       if image[x, y] == orig_value:
           image[x, y] = fill_value

           if x > 0:
             stack.add((x - 1, y))
           if x < (xsize - 1):
             stack.add((x + 1, y))
           if y > 0:
             stack.add((x, y - 1))
           if y < (ysize - 1):
             stack.add((x, y + 1))
