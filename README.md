# Unified Particle System for Multiple-fluid Flow and Porous Material

Bo Ren*, Ben Xu*(*joint first author)  Chenfeng Li

*ACM SIGGRAPH 2021*

## Abstract

Porous materials are common in daily life. They include granular
material (e.g. sand) that behaves like liquid flow when mixed with
fluid and foam material (e.g. sponge) that deforms like solid when
interacting with liquid. The underlying physics is further complicated when multiple fluids interact with porous materials involving
coupling between rigid and fluid bodies, which may follow different physics models such as the Darcy’s law and the multiple-fluid
Navier-Stokes equations. We propose a unified particle framework
for the simulation of multiple-fluid flows and porous materials. A
novel virtual phase concept is introduced to avoid explicit particle
state tracking and runtime particle deletion/insertion. Our unified
model is flexible and stable to cope with multiple fluid interacting with porous materials, and it can ensure consistent mass and
momentum transport over the whole simulation space.

## Paper  [[pdf]](http://ren-bo.net/papers/rb_multiporous2021.pdf)
## Video  [[mp4]](http://ren-bo.net/Videos/rb_multiporous2021.mp4)
## Description

On master branch, interaction between single-phase fluid and porous solid is implemented. 
On multifluids branch, multiple-fluids version is implemented.
## Requirements
This project is based on Fluids v.3 http://fluids3.com  We thank 2012 Hoetzlein, Rama for the great work. 
To run this code, please config the Fluids v.3 environment first and cover the source code.


![threefilter](https://github.com/BenXu86/DataSet/raw/main/threeFilter.gif)
![dumbbell](https://github.com/BenXu86/DataSet/raw/main/%E5%B0%8F%E7%90%83.gif)
![bunny](https://github.com/BenXu86/DataSet/raw/main/bunny.gif)
![S](https://github.com/BenXu86/DataSet/raw/main/S.gif)
