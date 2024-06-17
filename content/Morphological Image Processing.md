---
aliases:
  - Image Morphology
tags:
---
## Overview

Image Morphology is a comprehensive set of image processing operations that process images based on shapes. Morphological operations apply a structuring element to an input image then create an output image of the same size. 

In a morphological operation, the value of each pixel in the output is based on a comparison of the corresponding pixel in the input image with its neighbors.

There is a slight overlap between Image Morphology and [[Image Segmentation]]. Morphology consists of methods that can be used to pre-process the input data of image segmentation or to post-process its output image. In otherwords, once segmentation is complete, morphological operations can be used to remove imperfections in the segmented image and deliver information on the shape and structure of the image.

## Terminologies

- **Structuring Element**: it is a matrix or a small-sized template that is used to traverse an image. The structuring element is positioned at all possible locations in the image, and it is compared with the connected pixels. It can be of any shape.
- **Fit**: when all the pixels in the structuring element cover the pixels of the object, we call it fit
- **Hit**: when no pixel in the structuring element cover the pixels of the object, we call it miss.
![[static/images/morphological_terminologies.webp]]

## Operations
Fundamentally, morphological image processing is similar to [[Spatial Filtering]]. The structuring element is moved across every pixel in the original image to give a new pixel in a new processed image. The value of this new pixel depends on the morphological operation performed. The two most widely used operations are [[#Erosion]] and [[#Dilation]]

### Erosion
Erosion shrinks the image pixels, or erosion 

### Dilation
## Notes
### Resources
- 
### Further Readings
- [[Image Processing]]