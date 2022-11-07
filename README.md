# Poisson-blending

For seamless blending, image gradients are used instead of pixel intensity to blend two images by solving Poisson equations with a predefined boundary condition. Furthermore, this methodology ensures that the colour of the inserted image is also shifted, so that the inserted object appears to be part of the target image's environment. As a result, a bright object copied and pasted into a dark image will have its colour shifted to a darker colour.

![Screenshot 2022-11-06 at 8.22.17 PM.png]('Screenshot 2022-11-06 at 8.22.17 PM.png')
![Screenshot 2022-11-06 at 8.22.44 PM.png]('Screenshot 2022-11-06 at 8.22.44 PM.png')

