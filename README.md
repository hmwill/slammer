# slammer

Personal learning excursion into Simultanuous Location and Mapping (SLAM) methods. What's currently there 
is in the tradition of the ORB-SLAM family of systems. However, as a learning exercise, I decided to 
implement the algorithms from rather low above ground up. So, for example, there's my own implementation 
of ORB feature detection, description and matching on top of the Boost Generic Image Library (GIL) as image 
data structure, or a Pose and Places (PnP) optimizer using least squares just on top of linear algebra 
operations using the Eigen library.

I am focusing my efforts on RGB-D camera input, such as coming from an Intel RealSense camera. This is 
because I own such devices.

# Setup

I have been using CMake and Vcpkg to manage dependencies and build the the system. In addition, you will 
want to get a copy of the LORIS data set (see _Data Files_ below). 

__Warning:__ I am still in the middle of completing/reworking the initial version, so maybe hold off for a 
while until you try this code. Or shoot me an email.

## Data Files

In order to execute tests and example, a `data` folder needs to be created. Specifically, the folder should
contain the [OpenLORIS Scene Data Sets](https://shimo.im/docs/HhJj6XHYhdRQ6jjk).
