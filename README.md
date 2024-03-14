# overlapping_pixels

The idea is to represent an image with gaussians of colors. Each gaussian has then 8 attributes - 
`mean_x, mean_y, cov_xx, cov_xy, cov_yy, color_r, color_g, color_b`. The contribution of each gaussians 
at any pixel location are then summed up to obtain the final color at that pixel. e.g. the color of the pixel at location 
`r, c` is given by the following formula:

`\sum_{i=1}^{N} exp(([r, c] - mean_i)*cov_i^(-1)*([r, c] - mena_i)^T) * color_i

where `N` is the number of gaussians.

This way an image of resolution `128X128` can be expressd with just 400 gaussians. This is about 15X compression. The 
hope then is that this representation can be used by VITs that are much smaller to perform tasks like classification,
segmentation etc.

## Experiments
We converted the CelebA dataset to this representation and trained a VIT on it. The results are not promising. The 
vanilla VIT which is termed uniform VIT in the config file here achieves 90% validation accuracy. The VIT trained on 
gaussian representation achieves only 85.88% accuracy. This is a significant drop. The VIT trained on gaussian 
representation is able to overfir but seems to struggle to utilize the covariance information.